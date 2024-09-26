use ndarray::{Array1, Array2, Array3, Array4, ArrayD, Axis, ShapeBuilder};



#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ShapeBuilder;

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
fn snr_weigting(mask:&ArrayD<bool>,noise_std_map:&Array3<f32>) -> Array3<f32> {

    let mut w = 1./noise_std_map;

    // where mask is 0 or there are bad points, set w to 0

    let mut sum = 0.;
    let mut n = 0usize;

    w.iter_mut().zip(mask.iter()).for_each(|(x,&m)|{
        if !m {
            *x = 0.
        }
        if x.is_nan() || x.is_infinite() {
            *x = 0.;
        }

        if x.is_normal() {
            n += 1;
        }

        sum += *x;
    });

    let mean = sum / n as f32;

    return w / mean

}

/// discrete gradient operator over x. writes result to 4-D dst array where the last dimension is
/// Gx, Gy, Gz
fn fgrad_mut(x:&Array3<f32>,dst:&mut Array4<f32>, voxel_size:[f32;3]) {

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

    fgrad_mut(&x, &mut d,[1.,1.,1.]);

    assert_eq!(d.as_slice_memory_order().unwrap(),expected.as_slice());

}


/// returns a  mask of edges or its inverse (not sure)
fn gradient_weighting(magnitude:&Array3<f32>, mask:&Array3<bool>, voxel_size:[f32;3], edge_voxel_percentage:f32) -> Array4<bool> {

    // initialization to 1 % of max signal
    let mut field_noise_level = 0.01 * magnitude.iter().max_by(|a,b| a.partial_cmp(&b).unwrap()).unwrap();

    let magnitude = magnitude * mask.map(|&x| if x {1.0} else {0.});

    let dims = magnitude.shape();


    let mut grad = Array4::zeros((dims[0],dims[1],dims[2], 3).f());
    fgrad_mut(&magnitude, &mut grad, voxel_size);

    grad.mapv_inplace(|x| x.abs());

    let denominator = mask.iter().fold(0f32, |acc,&x| if x {acc + 1.} else {acc} );


    let grad_above_noise = |grad:&Array4<f32>,field_noise:f32| {
        grad.iter().fold(0f32, |acc,&x| {
            if x > field_noise {
                acc + 1.
            }else {
                acc
            }
        })
    };


    let mut numerator = grad_above_noise(&grad,field_noise_level);

    if numerator / denominator > edge_voxel_percentage {
        while numerator / denominator > edge_voxel_percentage {
            field_noise_level *= 1.05; // increaase by 5 %
            numerator = grad_above_noise(&grad,field_noise_level);
        }
    }else {
        while numerator / denominator < edge_voxel_percentage {
            field_noise_level *= 0.95; // decrease by 5 %
            numerator = grad_above_noise(&grad,field_noise_level);
        }
    }

    grad.map(|&x| x <= field_noise_level)

}

/*

function wG=gradient_mask(gradient_weighting_mode, iMag, Mask, grad, voxel_size, percentage)

if nargin < 6
    percentage = 0.9;
end


field_noise_level = 0.01*max(iMag(:));
wG = abs(grad(iMag.*(Mask>0), voxel_size));
denominator = sum(Mask(:)==1);
numerator = sum(wG(:)>field_noise_level);
if  (numerator/denominator)>percentage
    while (numerator/denominator)>percentage
        field_noise_level = field_noise_level*1.05;
        numerator = sum(wG(:)>field_noise_level);
    end
else
    while (numerator/denominator)<percentage
        field_noise_level = field_noise_level*.95;
        numerator = sum(wG(:)>field_noise_level);
    end
end

wG = (wG<=field_noise_level);

*/