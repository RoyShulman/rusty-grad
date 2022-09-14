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

#[derive(PartialEq)]
pub enum LossReduction {
    MEAN,
    SUM,
}

/// Apply cross entropy, which includes softmax
pub fn cross_entropy_loss(
    input: &[Vec<MutableScalarTensor>],
    target: &[Vec<f32>],
    reduce: LossReduction,
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

    if reduce == LossReduction::MEAN {
        sum = &sum / input.len() as f32;
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

        // Reduction should matter on a single tensor
        for reduction in [LossReduction::SUM, LossReduction::MEAN] {
            let real = cross_entropy_loss(&vec![input.clone()], &vec![target.clone()], reduction);
            assert!(
                (real.borrow().data - 2.7252).abs() < 0.0001,
                "real: {}, expected: {}",
                real.borrow().data,
                2.7252
            );
        }
    }

    #[test]
    fn test_cross_entropy_loss_sum() {
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

        let real = cross_entropy_loss(
            &vec![input1, input2],
            &vec![output1, output2],
            LossReduction::SUM,
        );
        assert!(
            (real.borrow().data - 6.2383).abs() < 0.001,
            "real: {}, expected: {}",
            real.borrow().data,
            6.2383
        );
    }

    #[test]
    fn test_cross_entropy_loss_mean() {
        let inputs = [
            [
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 5.2136, 0.0000, 2.4858, 0.0000, 2.5724,
            ],
            [
                1.4193, 0.0000, 0.3159, 0.0000, 3.8302, 4.9351, 0.0000, 0.0000, 1.2242, 0.0000,
            ],
            [
                9.4651, 0.0000, 3.6780, 0.0000, 7.3900, 6.7230, 0.0000, 0.0000, 0.7136, 9.9240,
            ],
            [
                4.2783, 0.0000, 0.0000, 0.0000, 2.6881, 4.4851, 0.0000, 6.4146, 0.3669, 4.2653,
            ],
            [
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 4.2967, 0.0000, 0.0000, 0.1937, 0.0000,
            ],
            [
                2.5722, 0.0000, 4.7754, 0.0000, 0.0000, 0.0000, 0.0000, 0.0840, 4.9622, 0.0000,
            ],
            [
                0.0000, 1.7514, 0.0000, 0.0000, 0.0000, 7.1554, 0.0000, 0.0000, 0.0000, 10.8839,
            ],
            [
                0.0000, 0.0000, 0.1193, 0.0000, 0.0000, 0.0000, 0.0000, 1.4858, 0.0000, 0.0000,
            ],
            [
                1.4163, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 7.1548, 0.0000,
            ],
            [
                0.6866, 1.1895, 6.6810, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 6.4898,
            ],
            [
                0.0000, 0.0000, 2.0479, 0.0000, 0.6218, 0.5415, 0.0000, 0.0000, 0.9339, 0.0000,
            ],
            [
                0.0000, 0.0000, 1.5827, 0.0000, 0.0000, 0.0000, 0.0000, 6.5913, 0.0000, 4.9947,
            ],
            [
                0.0000, 0.0000, 4.3675, 0.0000, 0.0000, 7.1172, 0.0000, 0.0000, 0.0000, 0.0000,
            ],
            [
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0198, 0.0000, 0.0000, 0.0000, 7.2033,
            ],
            [
                8.5282, 0.0000, 0.0485, 0.0000, 9.3667, 4.6131, 0.0000, 0.0000, 5.1600, 0.0000,
            ],
            [
                0.0000, 0.0000, 5.3622, 0.0000, 0.0000, 2.7705, 0.0000, 0.0000, 0.0000, 9.0917,
            ],
            [
                0.0000, 0.0000, 0.0000, 1.5794, 0.3976, 0.0000, 0.0000, 5.7793, 1.9515, 0.0000,
            ],
            [
                0.0000, 0.5745, 5.2741, 0.0000, 0.0000, 1.0341, 0.0000, 0.0000, 0.0000, 0.0000,
            ],
            [
                0.0000, 0.0000, 6.3219, 0.0000, 0.0000, 0.0000, 0.0000, 2.1995, 1.7168, 0.0000,
            ],
            [
                1.0281, 0.0000, 9.5872, 0.0000, 1.3884, 7.7158, 0.0000, 0.0000, 0.0000, 3.0671,
            ],
        ];
        let mut xs: Vec<Vec<MutableScalarTensor>> = vec![];
        for input in inputs {
            xs.push(input.iter().map(|x| ScalarTensor::new(*x)).collect());
        }

        let outputs = [
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        ];

        let mut ys: Vec<Vec<f32>> = vec![];
        for output in outputs {
            ys.push(output.iter().map(|x| *x).collect());
        }

        let real = cross_entropy_loss(&xs, &ys, LossReduction::MEAN);
        assert!(
            (real.borrow().data - 5.8185).abs() < 0.001,
            "{}",
            real.borrow().data
        );
    }
}
