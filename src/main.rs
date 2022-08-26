// #![deny(warnings)]
mod engine;
mod graph;
mod loss_functions;
mod nn;

use graph::ScalarTensorGraph;
use nn::MLP;

use crate::engine::MutableScalarTensor;

fn binary_classifier_example() {
    let n = MLP::new(3, vec![4, 4, 1]);
    let xs = vec![
        vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
    ];
    let ys = vec![1.0, -1.0, -1.0, 1.0];
    let ypreds = xs
        .iter()
        .map(|x| n.scalar_forward(x))
        .collect::<Vec<MutableScalarTensor>>();

    let loss = loss_functions::mse_loss(&ypreds, &ys);
    println!("{}", ScalarTensorGraph::draw_root(&loss));
}

fn main() {
    binary_classifier_example();
}
