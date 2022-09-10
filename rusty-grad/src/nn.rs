use crate::engine::{MutableScalarTensor, ScalarTensor};
use rand::{distributions::Uniform, thread_rng, Rng};

pub struct LinearNeuron {
    weights: Vec<MutableScalarTensor>,
    bias: MutableScalarTensor,
}

pub trait Module {
    fn zero_grad(&mut self);
    fn parameters(&self) -> Vec<MutableScalarTensor>;
    fn forward(&self, x: &Vec<MutableScalarTensor>) -> Vec<MutableScalarTensor>;
}

impl LinearNeuron {
    pub fn new(num_weights: usize) -> Self {
        let mut weights = Vec::new();
        let mut rng = thread_rng();
        let uniform = Uniform::new(-1.0, 1.0);
        for _ in 0..num_weights {
            weights.push(ScalarTensor::new(rng.sample(uniform)));
        }

        let bias = ScalarTensor::new(rng.sample(uniform));
        Self { weights, bias }
    }
}

impl Module for LinearNeuron {
    fn zero_grad(&mut self) {
        for w in self.weights.iter() {
            w.borrow_mut().zero_grad();
        }
        self.bias.borrow_mut().zero_grad();
    }

    fn parameters(&self) -> Vec<MutableScalarTensor> {
        let mut params = Vec::new();
        for w in self.weights.iter() {
            params.push(w.clone());
        }
        params.push(self.bias.clone());
        params
    }

    fn forward(&self, x: &Vec<MutableScalarTensor>) -> Vec<MutableScalarTensor> {
        if self.weights.len() != x.len() {
            panic!("Weights and input lengths must match");
        }

        // We can't use += because rust doesn't let us change the assigned to a different variable,
        // and we need to since it's now different with the added tensors being the children
        // so we use 2 of them
        let mut result = self.bias.clone(); // This is to avoid another node addition
                                            // w * x + b
        for (w_i, x_i) in self.weights.iter().zip(x.iter()) {
            result = result + w_i * x_i;
        }
        vec![result]
    }
}

pub struct LinearLayer {
    neurons: Vec<LinearNeuron>,
}

impl LinearLayer {
    pub fn new(num_inputs: usize, num_outputs: usize) -> Box<Self> {
        let mut neurons = Vec::new();
        for _ in 0..num_outputs {
            neurons.push(LinearNeuron::new(num_inputs));
        }
        Box::new(Self { neurons })
    }
}

impl Module for LinearLayer {
    fn zero_grad(&mut self) {
        for n in self.neurons.iter_mut() {
            n.zero_grad();
        }
    }

    fn parameters(&self) -> Vec<MutableScalarTensor> {
        let mut result = Vec::new();
        for n in self.neurons.iter() {
            result.extend(n.parameters());
        }
        result
    }

    fn forward(&self, x: &Vec<MutableScalarTensor>) -> Vec<MutableScalarTensor> {
        let mut result = Vec::new();
        for n in self.neurons.iter() {
            result.extend(n.forward(&x));
        }
        result
    }
}

struct ReluNeuron {}

impl Module for ReluNeuron {
    fn zero_grad(&mut self) {}

    fn parameters(&self) -> Vec<MutableScalarTensor> {
        vec![]
    }

    fn forward(&self, x: &Vec<MutableScalarTensor>) -> Vec<MutableScalarTensor> {
        // ReLU has the same output dimensions as input
        x.iter().map(|x| x.relu()).collect()
    }
}

pub struct ReluLayer {}

impl ReluLayer {
    pub fn new() -> Box<Self> {
        Box::new(Self {})
    }
}

impl Module for ReluLayer {
    fn zero_grad(&mut self) {}

    fn parameters(&self) -> Vec<MutableScalarTensor> {
        vec![]
    }

    fn forward(&self, x: &Vec<MutableScalarTensor>) -> Vec<MutableScalarTensor> {
        let mut result = Vec::new();
        for x_i in x.iter() {
            result.extend(ReluNeuron {}.forward(&vec![x_i.clone()]));
        }
        result
    }
}

pub struct MLP {
    layers: Vec<Box<dyn Module>>,
}

impl MLP {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self { layers }
    }

    pub fn scalar_forward(&self, x: &Vec<f32>) -> Vec<MutableScalarTensor> {
        let scalar_tensor_x = x.iter().map(|value| ScalarTensor::new(*value)).collect();
        self.forward(&scalar_tensor_x)
    }
}

impl Module for MLP {
    fn zero_grad(&mut self) {
        for l in self.layers.iter_mut() {
            l.zero_grad();
        }
    }

    fn parameters(&self) -> Vec<MutableScalarTensor> {
        let mut params = Vec::new();
        for l in self.layers.iter() {
            params.extend(l.parameters());
        }
        params
    }

    fn forward(&self, x: &Vec<MutableScalarTensor>) -> Vec<MutableScalarTensor> {
        // TODO: remove the skip somehow?
        let mut layer_result = self.layers[0].forward(x);
        for l in self.layers.iter().skip(1) {
            layer_result = l.forward(&layer_result);
        }
        layer_result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_forward() {
        let n = LinearNeuron::new(3);
        let out = n.forward(&vec![
            ScalarTensor::new(1.0),
            ScalarTensor::new(2.0),
            ScalarTensor::new(3.0),
        ]);
        let expected_result =
            &n.weights[0] * 1.0 + &n.weights[1] * 2.0 + &n.weights[2] * 3.0 + n.bias;
        {
            assert_eq!(1, out.len());
            let out = out[0].borrow();
            assert_eq!(expected_result.borrow().data, out.data);
        }
    }

    #[test]
    fn test_neuron_parameters() {
        let n = LinearNeuron::new(3);
        // 3 + 1 for the bias
        assert_eq!(4, n.parameters().len());
    }

    #[test]
    fn test_mlp() {
        let mlp = MLP::new(vec![
            LinearLayer::new(10, 3),
            LinearLayer::new(3, 2),
            LinearLayer::new(2, 1),
        ]);
        // 33 + 8 + 3
        assert_eq!(44, mlp.parameters().len());
    }

    #[test]
    fn test_linear_layer_forward() {
        let l = LinearLayer::new(2, 5);
        let result = l.forward(&vec![ScalarTensor::new(5.0), ScalarTensor::new(3.14)]);
        assert_eq!(5, result.len());
    }

    #[test]
    fn test_mlp_linear_layer_forward() {
        let mlp = MLP::new(vec![LinearLayer::new(2, 5), LinearLayer::new(5, 1)]);
        let result = mlp.scalar_forward(&vec![1.2, 3.5]);
        assert_eq!(1, result.len());
    }

    #[test]
    fn test_relu_neuron() {
        let n = ReluNeuron {};
        let result = n.forward(&vec![
            ScalarTensor::new(5.0),
            ScalarTensor::new(-2.123),
            ScalarTensor::new(220.0),
            ScalarTensor::new(3.14),
            ScalarTensor::new(-3.14),
        ]);
        // Output dimensions as input
        assert_eq!(5, result.len());
        for (expected, real) in vec![5.0, 0.0, 220.0, 3.14, 0.0].iter().zip(result.iter()) {
            assert_eq!(*expected, real.borrow().data);
        }
    }

    #[test]
    fn test_relu_layer_forward() {
        let l = ReluLayer {};
        let result = l.forward(&vec![ScalarTensor::new(5.0), ScalarTensor::new(3.14)]);
        assert_eq!(2, result.len());
    }

    #[test]
    fn test_mlp_linear_relu() {
        let mlp = MLP::new(vec![LinearLayer::new(5, 100), ReluLayer::new()]);
        let result = mlp.scalar_forward(&vec![5.0, 3.41, -1.2, -2.3, 22.2]);
        for r in result {
            assert!(r.borrow().data >= 0.0);
        }
    }
}
