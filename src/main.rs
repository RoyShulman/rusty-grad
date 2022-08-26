// #![deny(warnings)]
mod engine;
mod graph;
mod loss_functions;
mod nn;

use nn::{Module, MLP};

use crate::engine::MutableScalarTensor;

fn sgd(
    n: &mut MLP,
    inputs: &Vec<Vec<f32>>,
    target: &Vec<f32>,
    num_iterations: usize,
    learning_rate: f32,
) {
    for epoch in 0..num_iterations {
        // Forward pass
        let ypreds = inputs
            .iter()
            .map(|x| n.scalar_forward(x))
            .collect::<Vec<MutableScalarTensor>>();
        let loss = loss_functions::mse_loss(&ypreds, target);

        // Backward pass
        n.zero_grad();
        loss.backward();

        // Update params
        for p in n.parameters() {
            let mut p = p.borrow_mut();
            // We want the loss to go down so we add by negative grad
            p.data += -learning_rate * p.grad;
        }

        println!("epoch: {}, loss: {}", epoch, loss.borrow().data);
    }
    let ypreds = inputs
            .iter()
            .map(|x| n.scalar_forward(x))
            .collect::<Vec<MutableScalarTensor>>();
    for pred in ypreds.iter() {
        println!("{}", pred.borrow().data);
    }
}

fn binary_classifier_example() {
    let mut n = MLP::new(3, vec![4, 4, 1]);
    let xs = vec![
        vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
    ];
    let ys = vec![1.0, -1.0, -1.0, 1.0];
    sgd(&mut n, &xs, &ys, 1000, 0.01);
}

fn main() {
    binary_classifier_example();
}
