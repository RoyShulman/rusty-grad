#![deny(warnings)]

use rusty_grad::fit::sgd;
use rusty_grad::nn::{LinearLayer, MLP};

fn binary_classifier_example() {
    let mut n = MLP::new(vec![
        LinearLayer::new(3, 4),
        LinearLayer::new(4, 4),
        LinearLayer::new(4, 1),
    ]);
    let xs = vec![
        vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
    ];
    let ys = vec![1.0, -1.0, -1.0, 1.0];
    sgd(&mut n, &xs, &ys, 1000, 0.01, true);
}

fn main() {
    binary_classifier_example();
}
