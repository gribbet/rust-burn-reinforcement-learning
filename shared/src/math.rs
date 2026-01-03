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
    let sign = z
        .clone()
        .mask_fill(z.clone().lower_elem(0.0), -1.0)
        .mask_fill(z.clone().greater_elem(0.0), 1.0);

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

    let mut res = atan_z;

    // Add PI where x < 0
    let res_plus_pi = res.clone() + pi;
    res = res.mask_where(x_neg.clone(), res_plus_pi);

    // If y < 0 and we added PI (meaning x < 0), we are at > PI/2.
    // We want -PI relative to the original, or subtract 2PI from current.
    // If x < 0, y < 0 -> atan(z) > 0. res = atan(z) + PI > PI.
    // We want atan(z) - PI.
    // So subtract 2PI.

    let res_y_neg = res.clone().mask_where(res.clone().greater_elem(pi_2), res.clone() - 2.0 * pi);

    res = res.mask_where(y_neg, res_y_neg);

    // Handle x = 0
    let res_x_zero = Tensor::zeros_like(&res)
        .mask_fill(y.clone().greater_elem(0.0), pi_2)
        .mask_fill(y.clone().lower_elem(0.0), -pi_2);

    res = res.mask_where(x_zero, res_x_zero);

    res
}
