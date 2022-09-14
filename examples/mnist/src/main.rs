#![deny(warnings)]
mod dataset_utils;
mod download_utils;
extern crate flate2;

use std::{error::Error, path::Path};

use dataset_utils::{get_training_set, LabeledImage};
use download_utils::{download_dataset, download_testset};
use rand::Rng;
use rusty_grad::{
    engine::MutableScalarTensor,
    loss_functions::{self, plot_loss, EpochLoss, LossReduction},
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
        let loss = loss_functions::cross_entropy_loss(
            &ypreds,
            &target[num..num + mini_batch_size],
            LossReduction::MEAN,
        );

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

fn normalize_input(images: &Vec<LabeledImage>) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
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

    (inputs, outputs)
}

fn get_index_of_max_tensor(it: &[MutableScalarTensor]) -> usize {
    let mut max = 0.0;
    let mut index = 0;
    for (i, prob) in it.iter().enumerate() {
        let prob = prob.borrow().data;
        if prob > max {
            max = prob;
            index = i;
        }
    }
    index
}

fn get_index_of_max_f32(it: &[f32]) -> usize {
    let mut max = 0.0;
    let mut index = 0;
    for (i, prob) in it.iter().enumerate() {
        if *prob > max {
            max = *prob;
            index = i;
        }
    }
    index
}

fn evaluate(mlp: &MLP, inputs: &[Vec<f32>], outputs: &[Vec<f32>]) {
    let mut correct = 0;
    for (xs, ys) in inputs.iter().zip(outputs) {
        let result = mlp.scalar_forward(xs);
        if get_index_of_max_tensor(&result) == get_index_of_max_f32(&ys) {
            correct += 1;
        }
    }

    println!(
        "total: {}, correct: {}, percent correct: {}",
        inputs.len(),
        correct,
        correct as f32 / inputs.len() as f32
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    // Train
    let dataset = download_dataset(Path::new("./dataset"))?;
    let images = get_training_set(&dataset)?;
    let mut mlp = MLP::new(vec![
        LinearLayer::new(28 * 28, 8),
        ReluLayer::new(),
        LinearLayer::new(8, 8),
        ReluLayer::new(),
        LinearLayer::new(8, 10),
        ReluLayer::new(),
    ]);

    let (inputs, outputs) = normalize_input(&images);
    sgd(&mut mlp, &inputs, &outputs, 1000, 0.01, 20, true);

    // Evaluate
    let testset = download_testset(Path::new("./testset"))?;
    let test_images = get_training_set(&testset)?;
    let (test_inputs, test_outputs) = normalize_input(&test_images);
    evaluate(&mlp, &test_inputs, &test_outputs);

    Ok(())
}
