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

pub fn softmax(input: &[MutableScalarTensor]) -> Vec<MutableScalarTensor> {
    let mut result = vec![];
    let mut sum = ScalarTensor::new(0.0);
    for x_i in input.iter() {
        sum = sum + x_i.exp();
    }

    for x_i in input.iter() {
        result.push(&x_i.exp() / &sum);
    }
    result
}

/// Apply cross entropy, which includes softmax
pub fn cross_entropy_loss_sum(
    input: &[Vec<MutableScalarTensor>],
    target: &[Vec<f32>],
) -> MutableScalarTensor {
    if input.len() != target.len() {
        panic!("Input and target dimensions must match");
    }

    let mut sum = ScalarTensor::new(0.0);
    for (x_i, y_i) in input.iter().zip(target) {
        let normalized = softmax(x_i);
        for (x_j, y_j) in normalized.iter().zip(y_i) {
            sum = &sum + &(&x_j.ln() * -*y_j);
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let input = vec![
            ScalarTensor::new(-0.0654),
            ScalarTensor::new(-1.1302),
            ScalarTensor::new(0.8120),
            ScalarTensor::new(0.9402),
        ];
        let expected = vec![
            ScalarTensor::new(0.1542),
            ScalarTensor::new(0.0532),
            ScalarTensor::new(0.3709),
            ScalarTensor::new(0.4217),
        ];

        let normalized = softmax(&input);

        for (x_i, y_i) in normalized.iter().zip(expected) {
            let x = x_i.borrow().data;
            let y = y_i.borrow().data;
            assert!((x - y).abs() < 0.001, "{}, {}", x, y);
        }
    }

    #[test]
    fn test_cross_entropy_loss_single_tensor() {
        let input = vec![
            ScalarTensor::new(0.8938),
            ScalarTensor::new(-0.7245),
            ScalarTensor::new(0.2932),
            ScalarTensor::new(0.9960),
            ScalarTensor::new(1.8179),
            ScalarTensor::new(0.2327),
            ScalarTensor::new(-0.3215),
            ScalarTensor::new(0.9681),
            ScalarTensor::new(0.2355),
            ScalarTensor::new(-1.4423),
        ];
        let target = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

        let real = cross_entropy_loss_sum(&vec![input], &vec![target]);
        assert!(
            (real.borrow().data - 2.7252).abs() < 0.0001,
            "real: {}, expected: {}",
            real.borrow().data,
            2.7252
        );
    }

    #[test]
    fn test_cross_entropy_loss_sum_tensors() {
        let input1 = vec![
            ScalarTensor::new(-0.5384),
            ScalarTensor::new(-0.7928),
            ScalarTensor::new(-0.2381),
            ScalarTensor::new(1.4723),
            ScalarTensor::new(0.8172),
            ScalarTensor::new(0.1298),
            ScalarTensor::new(0.7332),
            ScalarTensor::new(0.4451),
            ScalarTensor::new(-1.1762),
            ScalarTensor::new(0.6778),
        ];
        let input2 = vec![
            ScalarTensor::new(0.6386),
            ScalarTensor::new(-0.8578),
            ScalarTensor::new(-0.9771),
            ScalarTensor::new(0.8304),
            ScalarTensor::new(-1.4483),
            ScalarTensor::new(-0.1740),
            ScalarTensor::new(2.8277),
            ScalarTensor::new(-0.3276),
            ScalarTensor::new(0.6212),
            ScalarTensor::new(-0.3592),
        ];

        let output1 = vec![0., 0., 0., 0., 0., 1., 0., 0., 0., 0.];
        let output2 = vec![0., 0., 0., 0., 0., 0., 0., 0., 0., 1.];

        let real = cross_entropy_loss_sum(&vec![input1, input2], &vec![output1, output2]);
        assert!(
            (real.borrow().data - 6.2383).abs() < 0.001,
            "real: {}, expected: {}",
            real.borrow().data,
            6.2383
        );
    }
}
