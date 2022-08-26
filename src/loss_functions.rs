use crate::engine::{MutableScalarTensor, ScalarTensor};

pub fn mse_loss(input: &Vec<MutableScalarTensor>, target: &Vec<f32>) -> MutableScalarTensor {
    if input.len() != target.len() {
        panic!("Input and target dimensions must match");
    }

    let mut loss = ScalarTensor::new(0.0); // Maybe remove the useless 0 node?
    for (input_i, target_i) in input.iter().zip(target) {
        loss = &loss + &(input_i - *target_i).pow(2);
    }
    loss
}
