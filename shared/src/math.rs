use burn::prelude::*;

// Approximation of atan(z) for z in [-1, 1]
// Polynomial: z * (0.99997726 - 0.33262347 * z^2 + 0.19354346 * z^4 - 0.11643287 * z^6 + 0.05265332 * z^8 - 0.01172120 * z^10)
fn approx_atan_core<B: Backend>(z: Tensor<B, 3>) -> Tensor<B, 3> {
    let z2 = z.clone() * z.clone();
    let z4 = z2.clone() * z2.clone();
    let z6 = z4.clone() * z2.clone();
    let z8 = z4.clone() * z4.clone();
    let z10 = z8.clone() * z2.clone();

    let poly = z10 * -0.01172120
        + z8 * 0.05265332
        + z6 * -0.11643287
        + z4 * 0.19354346
        + z2 * -0.33262347
        + 0.99997726;

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

    // Standard atan2 logic:
    // x > 0: atan(y/x)
    // x < 0: atan(y/x) + sign(y) * pi
    // x = 0: sign(y) * pi/2

    let y_sign = y.clone().sign();
    let offset = y_sign.clone() * pi;
    let mut res = atan_z.clone().mask_where(x_neg, atan_z + offset);

    let res_x_zero = y_sign * pi_2;
    res = res.mask_where(x_zero, res_x_zero);

    res
}
