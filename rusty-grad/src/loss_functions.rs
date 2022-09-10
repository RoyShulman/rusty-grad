use plotters::{
    prelude::{BitMapBackend, ChartBuilder, IntoDrawingArea, LabelAreaPosition},
    series::LineSeries,
    style::{GREEN, WHITE},
};

use crate::engine::{MutableScalarTensor, ScalarTensor};

pub struct EpochLoss {
    pub epoch: usize,
    pub loss: f32,
}

impl From<&EpochLoss> for (usize, f32) {
    fn from(e: &EpochLoss) -> Self {
        let EpochLoss { epoch, loss } = e;
        (*epoch, *loss)
    }
}

pub fn plot_loss(epoch_loss: Vec<EpochLoss>) {
    let max_epoch = epoch_loss.iter().map(|x| x.epoch).max().unwrap();
    let max_loss = epoch_loss.iter().map(|x| x.loss).reduce(f32::max).unwrap();
    let root_area = BitMapBackend::new("loss.png", (1920, 1080)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Loss", ("sans-serif", 40))
        .build_cartesian_2d(0..max_epoch, 0.0..(max_loss + 1.0))
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    ctx.draw_series(LineSeries::new(
        epoch_loss.iter().map(|point| point.into()),
        &GREEN,
    ))
    .unwrap();
}

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
