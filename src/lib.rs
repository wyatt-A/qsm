use std::{path::Path, time::Instant};

use ndarray::{Array1, Array2, Array3, Array4, ArrayD, AssignElem, Axis, Ix3, ShapeBuilder};
use ndarray_linalg::Scalar;
use nifti::{writer::WriterOptions, IntoNdArray, NiftiObject, ReaderOptions};
use num_complex::{Complex32};


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


    #[test]
    fn medi_prototype() {

        let matrix_size = [400,342,341];
        let voxel_size = [1f32,1f32,1f32];
        let lambda = 100.;
        let delta_te = 5e-3; // 5 ms
        let center_freq = 300e6; // 300 MHz
        let tol_norm_ratio = 0.1;
        let max_iter = 10;
        let cg_tol = 0.01;
        let cg_max_iter = 10;

        let mut noise_std = Array3::<f32>::zeros(matrix_size.f());
        let mask = Array3::<bool>::from_elem(matrix_size.f(),true);
        let rdf = Array3::<f32>::zeros(matrix_size.f());
        let magnitude = Array3::<f32>::zeros(matrix_size.f());

        // dc at the origin
        let dipole_kern = fft::conv_kernels::dipole_kernel_3d(matrix_size, [0.,0.,1.], voxel_size);

        // apply mask to noise array
        noise_std *= &mask.map(|&x| if x {1.} else {0.});

        let m = snr_weigting(&mask, &noise_std);
        let b0 = rdf.map(|&phase| Complex32::from_polar(1., phase)) * &m;
        let w_g = gradient_weighting(&magnitude, &mask, voxel_size, 0.9);

        let mut iter = 0;
        // iter=0;

        // primary iterate for chi
        let mut x = Array3::<f32>::zeros(matrix_size.f());

        let mut fft_tmp = ArrayD::<Complex32>::zeros(matrix_size.as_slice().f());
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


        let result = x;

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

//fidelity = @(dx)Dconv(conj(w).*w.*Dconv(dx) );
fn fidelity(dx:&Array3<f32>,w:&Array3<Complex32>,dipole_kern:&Array3<Complex32>) -> Array3<f32> {
    dipole_conv(
        &(w.map(|x|x.conj()) * w * dipole_conv(dx,dipole_kern)).map(|x|x.re()),
        dipole_kern
    )
}

// Dconv = @(dx) real(ifftn(D.*fftn(dx)));
fn dipole_conv(x:&Array3<f32>,dipole_kern:&Array3<Complex32>) -> Array3<f32> {
    let mut x = x.map(|&x|Complex32::from_real(x)).into_dyn();
    fft::fftn(&mut x);
    x *= dipole_kern;
    fft::ifftn(&mut x);
    x.map(|x|x.re()).into_dimensionality::<Ix3>().unwrap()
}

fn calc_vr(x:&Array3<f32>,w_g:&Array4<bool>,matrix_size:[usize;3],voxel_size:[f32;3],eps:f32) -> Array4<f32> {
    let mut tmp = Array4::<f32>::zeros((matrix_size[0],matrix_size[1],matrix_size[2],3).f());
    tmp.mapv_inplace(|_| 0.);
    fgrad(&x.map(|x|x.re()), &mut tmp, voxel_size);
    apply_mask4d(&mut tmp, w_g);
    tmp.mapv_inplace(|x| 1. / (x.abs().powi(2) + eps).sqrt());
    return tmp;
}

fn calc_w(x:&Array3<f32>,m:&Array3<f32>,dipole_kern:&Array3<Complex32>) -> Array3<Complex32> {
    let mut tmp = ArrayD::<Complex32>::zeros(x.shape());
    tmp.assign(&x.map(|&x|Complex32::from_real(x)));
    fft::fftn(&mut tmp);
    tmp *= dipole_kern;
    fft::ifftn(&mut tmp);
    (&tmp.map(|&x| (x * Complex32::I).exp()) * m).into_dimensionality::<Ix3>().unwrap()
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
    *x *= &mask.map(|&x| if x {1.} else {0.});
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
        let ap = a(&p);
        delta = normsq3(&r);
        let alpha = delta / dot3(&p,&ap);
        x = x + alpha * &p;
        r = &r - alpha * &ap;
        let beta = normsq3(&r) / delta;
        p = &r + beta * p;
        k += 1;
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
fn fgrad(x:&Array3<f32>,dst:&mut Array4<f32>, voxel_size:[f32;3]) {

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

    dst.mapv_inplace(|_|0.);

    for axis_idx in 0..3 {
    
        let g = grad.index_axis(Axis(3), axis_idx);

        dst.lanes_mut(Axis(axis_idx)).into_iter().zip(g.lanes(Axis(axis_idx))).for_each(|(mut x,y)|{

            y.iter().enumerate().zip(x.iter_mut()).for_each(|((i,_),x)|{

                // Dirichlet boundary condition
                let bdiv = if i == 0 {
                    y[i]
                }else if i == y.len() - 1 {
                    0. - y[i-1]
                }else {
                    y[i] - y[i-1]
                };

                *x += bdiv / voxel_size[axis_idx];

            });

        });

        dst.mapv_inplace(|x| -x);

    }

}

#[test]
fn test_bdiv() {
    let magnitude = read_nifti_f32("/Users/Wyatt/scratch/qsm_work/test_inputs/magnitude.nii");
    let dims = magnitude.shape().to_vec();
    let mut result = Array3::<f32>::zeros((dims[0],dims[1],dims[2]).f());
    let mut grad = Array4::zeros((dims[0],dims[1],dims[2], 3).f());
    fgrad(&magnitude, &mut grad, [1.,1.,1.]);
    write_nifti4d("/Users/Wyatt/scratch/qsm_work/test_inputs/rust_outputs/fgrad",&grad);
    bdiv(&grad, [1.,1.,1.], &mut result);
    write_nifti("/Users/Wyatt/scratch/qsm_work/test_inputs/rust_outputs/bdiv",&result);
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

    let denominator = mask.iter().fold(0f32, |acc,&x| if x {acc + 1.} else {acc} );

    let grad_above_noise = |grad:&Array4<f32>,field_noise:f32| {
        grad.map(|&x| if x > field_noise {1.} else {0.}).sum()
    };

    let mut iter = 0;
    let mut numerator = grad_above_noise(&grad,field_noise_level);

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
    let magnitude = read_nifti_f32("/Users/Wyatt/scratch/qsm_work/test_inputs/magnitude.nii");
    let mask = read_nifti_f32("/Users/Wyatt/scratch/qsm_work/test_inputs/mask_eroded.nii");
    let mask = mask.map(|&x| x.is_normal());
    let voxel_size = [1.,1.,1.];
    let now = Instant::now();
    let w_g = gradient_weighting(&magnitude,&mask,voxel_size,0.9);
    let dur = now.elapsed().as_millis();

    let w_g = w_g.map(|&x| if x {1f32} else {0.});
    
    write_nifti4d("/Users/Wyatt/scratch/qsm_work/test_inputs/rust_outputs/w_g",&w_g);
    println!("took {} ms",dur);
}




fn read_nifti_f32(nifti_base:impl AsRef<Path>) -> Array3<f32> {
    let nii = ReaderOptions::new();
    let vol = nii.read_file(nifti_base.as_ref()).unwrap().into_volume();
    vol.into_ndarray::<f32>().unwrap().into_dimensionality::<Ix3>().unwrap()
}

fn write_nifti(nifti_base:impl AsRef<Path>,vol:&Array3<f32>) {
    let nii = WriterOptions::new(nifti_base.as_ref());
    nii.write_nifti(&vol).expect("trouble writing to nifti");
}

fn write_nifti4d(nifti_base:impl AsRef<Path>,vol:&Array4<f32>) {
    let nii = WriterOptions::new(nifti_base.as_ref());
    nii.write_nifti(&vol).expect("trouble writing to nifti");
}