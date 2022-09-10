use crate::{
    engine::MutableScalarTensor,
    loss_functions::{self, plot_loss, EpochLoss},
    nn::{Module, MLP},
};

pub fn sgd(
    n: &mut MLP,
    inputs: &Vec<Vec<f32>>,
    target: &Vec<f32>,
    num_iterations: usize,
    learning_rate: f32,
    should_plot: bool,
) {
    let mut epoch_loss = Vec::new();

    for epoch in 0..num_iterations {
        // Forward pass
        let ypreds = inputs
            .iter()
            .map(|x| n.scalar_forward(x).remove(0))
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
