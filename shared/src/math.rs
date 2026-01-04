use burn::prelude::*;

// 5th-order approximation of atan(z) for z in [-1, 1] (4th order polynomial)
// z * (1.0 - 0.333 * z^2 + 0.193 * z^4)
fn approx_atan_core<B: Backend>(z: Tensor<B, 3>) -> Tensor<B, 3> {
    let z2 = z.clone() * z.clone();
    let z4 = z2.clone() * z2.clone();
    let poly = z4 * 0.193 + z2 * -0.333 + 1.0;
    z * poly
}

pub fn approx_atan<B: Backend>(z: Tensor<B, 3>) -> Tensor<B, 3> {
    let mask = z.clone().abs().greater_elem(1.0);
    let z_safe = z.clone().mask_fill(mask.clone().bool_not(), 1.0);
    let z_inv = 1.0 / z_safe;
    let z_core = z.clone().mask_where(mask.clone(), z_inv);

    let res = approx_atan_core(z_core);

    let pi_2 = std::f32::consts::PI / 2.0;
    let sign = z.clone().sign();

    let res_inv = sign * pi_2 - res.clone();

    res.mask_where(mask, res_inv)
}

pub fn approx_atan2<B: Backend>(y: Tensor<B, 3>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    let x_zero = x.clone().equal_elem(0.0);
    let x_safe = x.clone().mask_fill(x_zero.clone(), 1.0);
    let z = y.clone() / x_safe;

    let atan_z = approx_atan(z);

    let pi = std::f32::consts::PI;
    let pi_2 = pi / 2.0;

    let x_neg = x.clone().lower_elem(0.0);
    let y_neg = y.clone().lower_elem(0.0);

    // x < 0: atan(y/x) + (y < 0 ? -pi : pi)
    let offset = y_neg.clone().float().mul_scalar(-2.0).add_scalar(1.0) * pi;
    let mut res = atan_z.clone().mask_where(x_neg, atan_z + offset);

    // x = 0: sign(y) * pi/2
    let y_sign = y.clone().sign();
    let res_x_zero = y_sign * pi_2;
    res = res.mask_where(x_zero, res_x_zero);

    res
}
