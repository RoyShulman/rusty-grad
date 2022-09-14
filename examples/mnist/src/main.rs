#![deny(warnings)]
mod dataset_utils;
mod download_utils;
extern crate flate2;

use std::{error::Error, path::Path};

use dataset_utils::get_training_set;
use download_utils::download_dataset;
use rand::Rng;
use rusty_grad::{
    engine::MutableScalarTensor,
    loss_functions::{self, plot_loss, EpochLoss},
    nn::{LinearLayer, Module, ReluLayer, MLP},
};

fn sgd(
    n: &mut MLP,
    inputs: &Vec<Vec<f32>>,
    target: &Vec<Vec<f32>>,
    num_iterations: usize,
    learning_rate: f32,
    mini_batch_size: usize,
    should_plot: bool,
) {
    let mut epoch_loss = Vec::new();

    for epoch in 0..num_iterations {
        let num = rand::thread_rng().gen_range(0..inputs.len() - mini_batch_size);

        // Forward pass
        let ypreds = inputs[num..num + mini_batch_size]
            .iter()
            .map(|x| n.scalar_forward(x))
            .collect::<Vec<Vec<MutableScalarTensor>>>();
        let loss =
            loss_functions::cross_entropy_loss_sum(&ypreds, &target[num..num + mini_batch_size]);

        // Backward pass
        n.zero_grad();
        loss.backward();

        // Update params
        for p in n.parameters() {
            let mut p = p.borrow_mut();
            // We want the loss to go down so we add by negative grad
            p.data += -learning_rate * p.grad;
        }

        epoch_loss.push(EpochLoss {
            epoch,
            loss: loss.borrow().data,
        });
        println!("epoch: {}, loss: {}", epoch, loss.borrow().data);
    }

    if should_plot {
        // Graph the loss function
        plot_loss(epoch_loss);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let dataset = download_dataset(Path::new("./dataset"))?;
    let images = get_training_set(&dataset)?;
    let mut mlp = MLP::new(vec![
        LinearLayer::new(28 * 28, 16),
        ReluLayer::new(),
        // LinearLayer::new(16, 16),
        // ReluLayer::new(),
        LinearLayer::new(16, 10),
        ReluLayer::new(),
    ]);

    let mut inputs = vec![];
    let mut outputs = vec![];

    for image in images.iter() {
        // Normalize input
        inputs.push(
            image
                .pixels
                .pixels
                .iter()
                .map(|x| *x as f32 / 255.0)
                .collect(),
        );
        let mut current_label = vec![];
        for index in 0..10 {
            if image.label == index {
                current_label.push(1.0);
            } else {
                current_label.push(0.0);
            }
        }
        outputs.push(current_label);
    }
    sgd(&mut mlp, &inputs, &outputs, 1000, 0.001, 20, true);

    Ok(())
}
