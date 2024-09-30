use std::{path::Path, time::Instant};

use fft::conv_kernels::dipole_kernel_3d;
use ndarray::{concatenate, parallel::prelude::{IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator}, s, Array1, Array2, Array3, Array4, ArrayD, AssignElem, Axis, Ix3, Ix4, ShapeBuilder};
use ndarray_linalg::Scalar;
use nifti::{writer::WriterOptions, IntoNdArray, NiftiObject, ReaderOptions};
use num_complex::{Complex32};
use rayon::slice::ParallelSliceMut;
use rustfft::FftPlanner;


#[cfg(test)]
mod tests {
    use core::f32;
    use std::f32::consts::PI;

    use super::*;
    use ndarray::{Ix3, ShapeBuilder};
    use ndarray_linalg::Scalar;
    use num_complex::Complex;

    #[test]
    fn it_works() {
        // some noise added
    
        let b = Array1::from_vec(vec![39.5327,0.4491,108.8172,5.1854]);
        let x_gt = Array1::from_vec(vec![1.,2.,3.,4.]);
        let x0 = Array1::from_vec(vec![0.,0.,0.,0.]);

        let a_entries = vec![9.49795,-0.951941,12.6686,-1.51682,-0.951941,2.11705,-0.855753,-0.0664434,12.6686,-0.855753,31.5077,0.834279,-1.51682,-0.0664434,0.834279,1.08306];
        let a = MatrixOp {
            a:Array2::from_shape_vec((4,4), a_entries).unwrap()
        };
        cgsolve(&a, &b, x0, 0.);
    }

    //cargo test --package qsm --lib -- tests::medi_prototype --exact --nocapture
    #[test]
    fn medi_prototype() {

        let matrix_size = [590,360,360];
        let voxel_size = [1f32,1f32,1f32];
        let lambda = 1000.;
        let delta_te = 5e-3; // 5 ms
        let center_freq = 300e6; // 300 MHz
        let tol_norm_ratio = 0.1;
        let max_iter = 1;
        let cg_tol = 0.01;
        let cg_max_iter = 5;

        let magnitude = read_nifti_f32("/home/wyatt/test_data/qsm_test/magnitude.nii");
        let mask = read_nifti_f32("/home/wyatt/test_data/qsm_test/mask_eroded.nii").map(|x|x.is_normal());
        let rdf = read_nifti_f32("/home/wyatt/test_data/qsm_test/relative_difference_field3.nii");
        let mut noise_std = read_nifti_f32("/home/wyatt/test_data/qsm_test/error_estimate.nii");

        // dc at the origin
        let dipole_kern = fft::conv_kernels::dipole_kernel_3d(matrix_size, [0.,0.,1.], voxel_size);

        // apply mask to noise array
        apply_mask3d(&mut noise_std, &mask);

        let m = snr_weigting(&mask, &noise_std);
        let b0 = rdf.map(|&phase| Complex32::from_polar(1., phase)) * &m;
        let w_g = gradient_weighting(&magnitude, &mask, voxel_size, 0.9);

        let mut iter = 0;
        // iter=0;

        // primary iterate for chi
        let mut x = Array3::<f32>::zeros(matrix_size.f());

        // x = zeros(matrix_size); %real(ifftn(conj(D).*fftn((abs(m).^2).*RDF)));

        let mut res_norm_ratio = f32::INFINITY;
        // res_norm_ratio = Inf;

        let mut cost_data_history:Vec<f32> = vec![];
        let mut cost_reg_history:Vec<f32> = vec![];
        // cost_data_history = zeros(1,max_iter);
        // cost_reg_history = zeros(1,max_iter);
        
        let eps = 0.000001;
        // e=0.000001; %a very small number to avoid /0

        //let mut badpoint = Array3::<bool>::from_elem(matrix_size.f(), false);
        // badpoint = zeros(matrix_size);

        // let dipole_conv = |x:&Array3<f32>| {
        //     let mut x = x.map(|&x|Complex32::from_real(x)).into_dyn();
        //     fft::fftn(&mut x);
        //     x *= &dipole_kern;
        //     fft::ifftn(&mut x);
        //     x.map(|x|x.re()).into_dimensionality::<Ix3>().unwrap()
        // };
        // Dconv = @(dx) real(ifftn(D.*fftn(dx)));
        // optimization loop
        while (res_norm_ratio>tol_norm_ratio)&&(iter<max_iter) {
            iter += 1;

            //Vr = 1./sqrt(abs(wG.*grad(real(x),voxel_size)).^2+e);
            // calculate vr
            let vr = calc_vr(&x, &w_g, matrix_size, voxel_size, eps);

            //w = m.*exp(1i*ifftn(D.*fftn(x)));
            let w = calc_w(&x, &m, &dipole_kern);
            
            // forward linear operator for conjugate gradient solver
            let a = |dx:&Array3<f32>| {
                reg(&dx,&w_g,&vr,matrix_size,voxel_size) + 2. * lambda * fidelity(dx,&w,&dipole_kern)
            };

            // data vector for conjugate gradient solver
            let mut b = reg(&x,&w_g,&vr,matrix_size,voxel_size) + 2. * lambda * dipole_conv(
                &((&w - &b0).map(|x| x * Complex32::I.conj()) * w.map(|x|x.conj())).map(|x|x.re()),
                &dipole_kern
            );
            b.par_mapv_inplace(|x| - x);


            let cg_x0 = Array3::zeros(matrix_size);
            let dx = cg_solve(a,&b,cg_x0,cg_tol,cg_max_iter);

            res_norm_ratio = normsq3(&dx).sqrt()/normsq3(&x.map(|x|x.re())).sqrt();

            // take update step
            x.scaled_add(1., &dx);

            //wres=m.*exp(1i*(real(ifftn(D.*fftn(x))))) - b0;
            let w_res = calc_w_res(&x, &m, &b0, &dipole_kern);
            norm_complex(&w_res);
            
            cost_data_history.push(norm_complex(&w_res));
            cost_reg_history.push(calc_reg_cost(&x, &w_g, matrix_size, voxel_size));

        }

        // final conversion to ppm
        //x = x/(2*pi*delta_TE*CF)*1e6.*Mask;
        let scale = 1e6 / (2. * PI * delta_te * center_freq);
        x.mapv_inplace(|x|x * scale);


        write_nifti("/home/wyatt/test_data/qsm_test/rust_outputs/chi.nii", &x);

    }

}

//reg = @(dx) div(wG.*(Vr.*(wG.*grad(real(dx),voxel_size))),voxel_size);
fn reg(dx:&Array3<f32>,w_g:&Array4<bool>,vr:&Array4<f32>,matrix_size:[usize;3],voxel_size:[f32;3]) -> Array3<f32> {
    let mut bdiv_out = Array3::<f32>::zeros(matrix_size.f());
    let mut tmp = Array4::<f32>::zeros((matrix_size[0],matrix_size[1],matrix_size[2],3).f());
    fgrad(&dx, &mut tmp, voxel_size);
    apply_mask4d(&mut tmp, w_g);
    tmp *= vr;
    apply_mask4d(&mut tmp, w_g);
    bdiv(&tmp, voxel_size, &mut bdiv_out);
    bdiv_out
}

#[test]
fn test_reg() {

    let matrix_size = [590,360,360];
    let voxel_size = [1.,1.,1.];
    let eps = 0.000001;

    let x = read_nifti_f32("/home/wyatt/test_data/qsm_test/x_test.nii");
    let w_g = read_nifti_f32_4d("/home/wyatt/test_data/qsm_test/w_g.nii").map(|x| x.is_normal());


    let vr = calc_vr(&x, &w_g, matrix_size, voxel_size, eps);

    let now = Instant::now();
    let reg = reg(&x,&w_g,&vr,matrix_size,voxel_size);
    let dur = now.elapsed().as_millis();

    println!("took {} ms",dur);

    write_nifti("/home/wyatt/test_data/qsm_test/rust_outputs/reg.nii", &reg);
}

//fidelity = @(dx)Dconv(conj(w).*w.*Dconv(dx) );
fn fidelity(dx:&Array3<f32>,w:&Array3<Complex32>,dipole_kern:&Array3<Complex32>) -> Array3<f32> {
    dipole_conv(
        &(w.map(|x|x.norm_sqr()) * dipole_conv(dx,dipole_kern)).map(|x|x.re()),
        dipole_kern
    )
}

#[test]
fn test_fidelity() {

    let x = read_nifti_f32("/home/wyatt/test_data/qsm_test/x_test.nii");
    let m = read_nifti_f32("/home/wyatt/test_data/qsm_test/m.nii");
    let d = dipole_kernel_3d([590,360,360], [0.,0.,1.], [1.,1.,1.]);
    let w = calc_w(&x, &m, &d);

    let now = Instant::now();
    let f = fidelity(&x, &w, &d);
    let dur = now.elapsed().as_millis();

    println!("took {} ms",dur);

    write_nifti("/home/wyatt/test_data/qsm_test/rust_outputs/fidel.nii", &f);

}


// Dconv = @(dx) real(ifftn(D.*fftn(dx)));
fn dipole_conv(x:&Array3<f32>,dipole_kern:&Array3<Complex32>) -> Array3<f32> {
    let mut x = x.map(|x|Complex32::from_real(*x)).into_dyn();
    fft::fftn(&mut x);
    x *= dipole_kern;
    fft::ifftn(&mut x);
    x.map(|x|x.re()).into_dimensionality::<Ix3>().unwrap()
}

fn calc_vr(x:&Array3<f32>,w_g:&Array4<bool>,matrix_size:[usize;3],voxel_size:[f32;3],eps:f32) -> Array4<f32> {
    let mut tmp = Array4::<f32>::zeros((matrix_size[0],matrix_size[1],matrix_size[2],3).f());
    tmp.mapv_inplace(|_| 0.);
    fgrad(&x, &mut tmp, voxel_size);
    apply_mask4d(&mut tmp, w_g);
    tmp.par_mapv_inplace(|x| 1. / (x.abs().powi(2) + eps).sqrt());
    //tmp.mapv_inplace(|x| 1. / (x.abs().powi(2) + eps).sqrt());
    return tmp;
}

#[test]
fn test_calc_vr() {

    let matrix_size = [590,360,360];
    let voxel_size = [1.,1.,1.];
    let eps = 0.000001;

    let x = read_nifti_f32("/home/wyatt/test_data/qsm_test/x_test.nii");
    let w_g = read_nifti_f32_4d("/home/wyatt/test_data/qsm_test/w_g.nii").map(|x| x.is_normal());
    
    let now = Instant::now();
    let vr = calc_vr(&x, &w_g, matrix_size, voxel_size, eps);
    let dur = now.elapsed().as_millis();

    println!("took {} ms",dur);

    write_nifti4d("/home/wyatt/test_data/qsm_test/rust_outputs/vr.nii", &vr);

}


fn calc_w(x:&Array3<f32>,m:&Array3<f32>,dipole_kern:&Array3<Complex32>) -> Array3<Complex32> {
    let mut tmp = x.map(|&x|Complex32::from_real(x)).into_dyn();
    let now = Instant::now();
    fft::fftn(&mut tmp);
    let dur = now.elapsed().as_millis();
    println!("fft took {} ms",dur);
    tmp *= dipole_kern;
    fft::ifftn(&mut tmp);
    tmp.par_map_inplace(|x|{
        *x = (*x * Complex32::I).exp()
    });
    m * tmp.into_dimensionality::<Ix3>().unwrap()
}

#[test]
fn calc_w_test() {
    let x = read_nifti_f32("/home/wyatt/test_data/qsm_test/x_test.nii");
    let m = read_nifti_f32("/home/wyatt/test_data/qsm_test/m.nii");
    let d = dipole_kernel_3d([590,360,360], [0.,0.,1.], [1.,1.,1.]);
    let now = Instant::now();
    let w = calc_w(&x, &m, &d);
    let dur = now.elapsed().as_millis();
    println!("took {} ms",dur);
    write_nifti("/home/wyatt/test_data/qsm_test/rust_outputs/w.nii", &w.map(|x|x.to_polar().1));
}

fn calc_w_res(x:&Array3<f32>,m:&Array3<f32>,b0:&Array3<Complex32>,dipole_kern:&Array3<Complex32>) -> Array3<Complex32> {
    let mut tmp = ArrayD::<Complex32>::zeros(x.shape());
    tmp.assign(&x.map(|&x|Complex32::from_real(x)));
    fft::fftn(&mut tmp);
    tmp *= dipole_kern;
    fft::ifftn(&mut tmp);
    m * tmp.map(|x|Complex32::new(0.,x.re()).exp()).into_dimensionality::<Ix3>().unwrap() - b0
}

// L1 regularization cost value
fn calc_reg_cost(x:&Array3<f32>,w_g:&Array4<bool>,matrix_size:[usize;3],voxel_size:[f32;3]) -> f32 {
    let mut tmp = Array4::<f32>::zeros((matrix_size[0],matrix_size[1],matrix_size[2],3).f());
    fgrad(&x, &mut tmp, voxel_size);
    apply_mask4d(&mut tmp, w_g);
    tmp.map(|x|x.abs()).into_iter().sum::<f32>()
}

fn apply_mask4d(x:&mut Array4<f32>, mask:&Array4<bool>) {
    //*x *= &mask.map(|&x| if x {1.} else {0.});
    x.as_slice_memory_order_mut().unwrap().par_iter_mut().zip(
        mask.as_slice_memory_order().unwrap().par_iter()
    ).for_each(|(x,&m)|{
        if !m {
            *x = 0.;
        }
    });
}

fn apply_mask3d(x:&mut Array3<f32>, mask:&Array3<bool>) {
    //*x *= &mask.map(|&x| if x {1.} else {0.});
    x.as_slice_memory_order_mut().unwrap().par_iter_mut().zip(
        mask.as_slice_memory_order().unwrap().par_iter()
    ).for_each(|(x,&m)|{
        if !m {
            *x = 0.;
        }
    });
}


struct MatrixOp {
    a:Array2<f32>
}

impl LinearOperator for MatrixOp {
    fn forward(&self,x:&Array1<f32>) -> Array1<f32> {
        self.a.dot(x)
    }
}

trait LinearOperator {
    fn forward(&self,x:&Array1<f32>) -> Array1<f32>;
}


fn cg_solve<A>(a:A,b:&Array3<f32>,x0:Array3<f32>,tol:f32,max_iter:usize) -> Array3<f32> 
where A: Fn(&Array3<f32>) -> Array3<f32>{
    
    let mut x = x0;
    let mut r = b - a(&x);

    let delta0 = normsq3(&b);
    let mut delta = normsq3(&r);
    // check if x0 is already the solution
    if delta <= tol * tol * delta0 {
        return x
    }

    let mut p = r.clone();

    let mut k = 0;

    while k < max_iter &&  delta > tol * tol * delta0 {
        println!("cg iter {}",k);
        let now = Instant::now();
        let ap = a(&p);
        delta = normsq3(&r);
        let alpha = delta / dot3(&p,&ap);
        x = x + alpha * &p;
        r = &r - alpha * &ap;
        let beta = normsq3(&r) / delta;
        p = &r + beta * p;
        k += 1;
        let dur = now.elapsed().as_millis();
        println!("cg iter took {} ms",dur);
    }
    return x
}



fn cgsolve<A:LinearOperator>(a:&A,b:&Array1<f32>,x0:Array1<f32>,tol:f32) {

    let max_iter = 10;

    let mut x = x0;
    let mut r = b - a.forward(&x);

    let delta0 = normsq(&b);
    let mut delta = normsq(&r);
    // check if x0 is already the solution
    if delta <= tol * tol * delta0 {
        return
    }

    println!("starting residual: {}",&r);

    let mut p = r.clone();

    let mut k = 0;

    while k < max_iter &&  delta > tol * tol * delta0 {
        let ap = a.forward(&p);
        delta = normsq(&r);
        let alpha = delta / dot(&p,&ap);
        x = x + alpha * &p;
        r = &r - alpha * &ap;
        println!("x: {}",&x);
        let beta = normsq(&r) / delta;
        p = &r + beta * p;
        k += 1;
    }
    println!("final x: {:?}",x);
}

fn normsq(x:&Array1<f32>) -> f32 {
    x.iter().map(|&x| x*x).sum::<f32>()
}

fn dot(a:&Array1<f32>,b:&Array1<f32>) -> f32 {
    a.iter().zip(b.iter()).map(|(&a,&b)| a * b).sum()
}

fn normsq3(x:&Array3<f32>) -> f32 {
    x.iter().map(|&x| x*x).sum::<f32>()
}

fn dot3(a:&Array3<f32>,b:&Array3<f32>) -> f32 {
    a.iter().zip(b.iter()).map(|(&a,&b)| a * b).sum()
}

fn norm_complex(a:&Array3<Complex32>) -> f32 {
    a.iter().map(|x|x.abs().powi(2)).sum::<f32>().sqrt()
}


/*

    x - 3D qsm iterate
    wG - 4D gradient mask thing
    e - small number to avoid div by 0
    grad - discrete gradient operator R3 -> R4
    div - discrete divergence operator R4 -> R3
    
    Vr = 1./sqrt(abs(wG.*grad(real(x),voxel_size)).^2+e);
    reg = @(dx) div(wG.*(Vr.*(wG.*grad(real(dx),voxel_size))),voxel_size);


    regularization is a function of the update step dx as well as the iterate x


*/


/// returns a weighting mask based on a noise map of the image, where the noise level is expressed as
/// the standard deviation
fn snr_weigting(mask:&Array3<bool>,noise_std_map:&Array3<f32>) -> Array3<f32> {
    let mask = mask.map(|&x| if x {1.} else {0.});
    let n = mask.sum();
    let mut w = mask/noise_std_map;
    w.mapv_inplace(|x|{
        if !x.is_finite() {
            0.
        }else {
            x
        }
    });
    let mean = w.sum() / n;
    return w / mean
}

#[test]
fn test_snr_weighting() {
    let mask = read_nifti_f32("/Users/Wyatt/scratch/qsm_work/test_inputs/mask_eroded.nii");
    let mask = mask.map(|x| x.is_normal());
    let error_map = read_nifti_f32("/Users/Wyatt/scratch/qsm_work/test_inputs/error_estimate.nii");
    let now = Instant::now();
    let m = snr_weigting(&mask,&error_map);
    let dur = now.elapsed().as_millis();
    write_nifti("/Users/Wyatt/scratch/qsm_work/test_inputs/rust_outputs/m", &m);
    println!("took {} ms",dur);
}


/// discrete gradient operator over x. writes result to 4-D dst array where the last dimension is
/// Gx, Gy, Gz
fn _fgrad(x:&Array3<f32>,dst:&mut Array4<f32>, voxel_size:[f32;3]) {

    // set dst to 0
    dst.mapv_inplace(|_| 0.);   

    for axis_idx in 0..3 {

        // get a mutable reference to the lanes of dest
        let mut dest_axis = dst.index_axis_mut(Axis(3), axis_idx);
        let dest_lanes = dest_axis.lanes_mut(Axis(axis_idx));
        let x_lanes = x.lanes(Axis(axis_idx));

        // apply forward difference over each lane and write to destination
        x_lanes.into_iter().zip(dest_lanes).for_each(|(x,mut y)|{
            // neuman boundary condition
            x.windows(2).into_iter().zip(y.iter_mut()).for_each(|(w,val)|{
                *val = w[1] - w[0] / voxel_size[axis_idx];
            });
        });
    }
}

fn fgrad(x:&Array3<f32>,dst:&mut Array4<f32>, voxel_size:[f32;3]) {

    let (mx,my,mz) = (x.shape()[0],x.shape()[1],x.shape()[2]);

    dst.slice_mut(s![0..mx-1,..,..,0])
    .assign(&(&x.slice(s![1..mx,..,..]) - &x.slice(s![0..mx-1,..,..])));
    dst.slice_mut(s![mx-1,..,..,0])
    .fill(0.);
    let mut gx = dst.slice_mut(s![..,..,..,0]);
    gx /= voxel_size[0];

    dst.slice_mut(s![..,0..my-1,..,1])
    .assign(&(&x.slice(s![..,1..my,..]) - &x.slice(s![..,0..my-1,..])));
    dst.slice_mut(s![..,my-1,..,1])
    .fill(0.);
    let mut gy = dst.slice_mut(s![..,..,..,1]);
    gy /= voxel_size[1];

    dst.slice_mut(s![..,..,0..mz-1,2])
    .assign(&(&x.slice(s![..,..,1..mz]) - &x.slice(s![..,..,0..mz-1])));
    dst.slice_mut(s![..,..,mz-1,2])
    .fill(0.);
    let mut gz = dst.slice_mut(s![..,..,..,2]);
    gz /= voxel_size[2];

}



#[test]
fn test_fgrad() {

    let x = vec![0.,1.,2.,3.,4.,5.,6.,7.];
    let expected = vec![1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,2.0,2.0,0.0,0.0,2.0,2.0,0.0,0.0,4.0,4.0,4.0,4.0,0.0,0.0,0.0,0.0];

    let x = Array3::from_shape_vec((2usize,2usize,2usize).f(), x).unwrap();
    
    let mut d = Array4::ones((2,2,2,3).f());

    fgrad(&x, &mut d,[1.,1.,1.]);

    assert_eq!(d.as_slice_memory_order().unwrap(),expected.as_slice());

}

/// backward divergence
fn bdiv(grad:&Array4<f32>,voxel_size:[f32;3],dst:&mut Array3<f32>) {

    // Extract Gx_x, Gx_y, Gx_z (these are 3D slices of the 4D array grad)
    let gx = grad.slice(s![.., .., .., 0]);
    let gy = grad.slice(s![.., .., .., 1]);
    let gz = grad.slice(s![.., .., .., 2]);

    let (mx, my, mz) = (gx.shape()[0], gx.shape()[1], gx.shape()[2]);

    // Initialize Dx, Dy, Dz
    let mut dx = Array3::<f32>::zeros((mx, my, mz).f());
    let mut dy = Array3::<f32>::zeros((mx, my, mz).f());
    let mut dz = Array3::<f32>::zeros((mx, my, mz).f());

    // Compute Dx: backward difference in the x direction
    dx.slice_mut(s![1..mx-1, .., ..])
        .assign(&(&gx.slice(s![1..mx-1,..,..,]) - &gx.slice(s![0..mx-2,..,..,])));
    dx.slice_mut(s![0,..,..]).assign(&gx.slice(s![0,..,..,]));
    dx.slice_mut(s![mx-1,..,..]).assign(&-&gx.slice(s![mx-2,..,..,]));
    dx /= voxel_size[0];

    // Compute Dy: backward difference in the y direction
    dy.slice_mut(s![.., 1..my-1, ..])
        .assign(&(&gy.slice(s![..,1..my-1,..,]) - &gy.slice(s![..,0..my-2,..,])));
    dy.slice_mut(s![..,0,..]).assign(&gy.slice(s![..,0,..,]));
    dy.slice_mut(s![..,my-1,..]).assign(&-&gy.slice(s![..,my-2,..,]));
    dy /= voxel_size[1];

    // Compute Dz: backward difference in the z direction
    dz.slice_mut(s![.., .., 1..mz-1])
        .assign(&(&gz.slice(s![..,..,1..mz-1]) - &gz.slice(s![..,..,0..mz-2])));
    dz.slice_mut(s![..,..,0]).assign(&gz.slice(s![..,..,0]));
    dz.slice_mut(s![..,..,mz-1]).assign(&-&gz.slice(s![..,..,mz-2]));
    dz /= voxel_size[2];

    // Compute the divergence and assign to dst
    dst.assign(&-(dx + dy + dz));

}


#[test]
fn test_bdiv() {
    let input = "/home/wyatt/test_data/qsm_test/matlab_outputs/fgrad.nii";
    let input = read_nifti_f32_4d(input);
    //println!("{:?}",input.slice(s![..,..,1,0]));
    let s = input.shape();
    let mut result = Array3::zeros((s[0],s[1],s[2]).f());
    let now = Instant::now();
    bdiv(&input, [1.,1.,1.], &mut result);
    let dur = now.elapsed().as_millis();
    write_nifti("/home/wyatt/test_data/qsm_test/rust_outputs/bdiv_out",&result);
    //println!("{:#?}",result.as_slice_memory_order().unwrap());
    println!("took {} ms",dur);
}


/// returns a  mask of edges or its inverse (not sure)
fn gradient_weighting(magnitude:&Array3<f32>, mask:&Array3<bool>, voxel_size:[f32;3], edge_voxel_percentage:f32) -> Array4<bool> {

    // initialization to 1 % of max signal
    let mut field_noise_level = 0.01 * magnitude.iter().max_by(|a,b| a.partial_cmp(&b).unwrap()).unwrap();

    println!("field noise level: {}",field_noise_level);

    let magnitude = magnitude * mask.map(|&x| if x {1.0} else {0.});

    let dims = magnitude.shape();

    let mut grad = Array4::zeros((dims[0],dims[1],dims[2], 3).f());

    let now = Instant::now();
    fgrad(&magnitude, &mut grad, voxel_size);
    let dur = now.elapsed().as_millis();
    println!("fgrad took: {} ms",dur);

    grad.mapv_inplace(|x| x.abs());

    //let denominator = mask.iter().fold(0f32, |acc,&x| if x {acc + 1.} else {acc} );
    let denominator = mask.as_slice_memory_order().unwrap()
    .par_iter().filter(|&&x| x).count() as f32;

    let grad_above_noise = |grad:&Array4<f32>,field_noise:f32| {
        grad.as_slice_memory_order()
        .unwrap()
        .par_iter()
        .filter(|&&x| x > field_noise)
        .count() as f32
    };

    let mut iter = 0;
    let now = Instant::now();
    let mut numerator = grad_above_noise(&grad,field_noise_level);
    let dur = now.elapsed().as_millis();
    println!("took: {} ms",dur);

    if numerator / denominator > edge_voxel_percentage {
        while numerator / denominator > edge_voxel_percentage {
            field_noise_level *= 1.05; // increaase by 5 %
            numerator = grad_above_noise(&grad,field_noise_level);
            iter += 1;
        }
    }else {
        while numerator / denominator < edge_voxel_percentage {
            field_noise_level *= 0.95; // decrease by 5 %
            numerator = grad_above_noise(&grad,field_noise_level);
            iter += 1;
        }
    }
    println!("n_iter = {}",iter);
    grad.map(|&x| x <= field_noise_level)
    
}

#[test]
fn test_gradient_weighting() {
    let magnitude = read_nifti_f32("/home/wyatt/test_data/qsm_test/magnitude.nii");
    let mask = read_nifti_f32("/home/wyatt/test_data/qsm_test/mask_eroded.nii");
    let mask = mask.map(|&x| x.is_normal());
    let voxel_size = [1.,1.,1.];
    let now = Instant::now();
    let w_g = gradient_weighting(&magnitude,&mask,voxel_size,0.9);
    let dur = now.elapsed().as_millis();

    let w_g = w_g.map(|&x| if x {1f32} else {0.});
    
    write_nifti4d("/home/wyatt/test_data/qsm_test/rust_outputs/w_g",&w_g);
    println!("took {} ms",dur);
}

#[test]
fn test_dipole_kernel() {

    let dk = fft::conv_kernels::dipole_kernel_3d([590,360,360], [0.,0.,1.], [1.,1.,1.]);
    write_nifti("/home/wyatt/test_data/qsm_test/rust_outputs/dipole_kern", &dk.map(|x|x.re()));

}



fn read_nifti_f32(nifti_base:impl AsRef<Path>) -> Array3<f32> {
    let nii = ReaderOptions::new();
    let vol = nii.read_file(nifti_base.as_ref()).unwrap().into_volume();
    vol.into_ndarray::<f32>().unwrap().into_dimensionality::<Ix3>().unwrap()
}

fn read_nifti_f32_4d(nifti_base:impl AsRef<Path>) -> Array4<f32> {
    let nii = ReaderOptions::new();
    let vol = nii.read_file(nifti_base.as_ref()).unwrap().into_volume();
    vol.into_ndarray::<f32>().unwrap().into_dimensionality::<Ix4>().unwrap()
}

fn write_nifti(nifti_base:impl AsRef<Path>,vol:&Array3<f32>) {
    let nii = WriterOptions::new(nifti_base.as_ref());
    nii.write_nifti(&vol).expect("trouble writing to nifti");
}

fn write_nifti4d(nifti_base:impl AsRef<Path>,vol:&Array4<f32>) {
    let nii = WriterOptions::new(nifti_base.as_ref());
    nii.write_nifti(&vol).expect("trouble writing to nifti");
}
