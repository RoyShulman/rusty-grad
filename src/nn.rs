use crate::engine::{MutableScalarTensor, ScalarTensor};
use rand::{distributions::Uniform, thread_rng, Rng};

pub struct Neuron {
    weights: Vec<MutableScalarTensor>,
    bias: MutableScalarTensor,
}

enum ForwardResult {
    Multi(Vec<MutableScalarTensor>),
    Single(MutableScalarTensor),
}

pub trait Module {
    fn zero_grad(&mut self);
    fn parameters(&self) -> Vec<MutableScalarTensor>;
}

impl Neuron {
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

    // TODO: Think about how to turn this into a trait
    pub fn forward(&self, x: &Vec<MutableScalarTensor>) -> MutableScalarTensor {
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
        result
    }
}

impl Module for Neuron {
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
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(num_inputs: usize, num_outputs: usize) -> Self {
        let mut neurons = Vec::new();
        for _ in 0..num_outputs {
            neurons.push(Neuron::new(num_inputs));
        }
        Self { neurons }
    }

    fn forward(&self, x: &Vec<MutableScalarTensor>) -> ForwardResult {
        let mut result = Vec::new();
        for n in self.neurons.iter() {
            result.push(n.forward(&x));
        }
        if result.len() == 1 {
            ForwardResult::Single(result.remove(0))
        } else {
            ForwardResult::Multi(result)
        }
    }
}

impl Module for Layer {
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
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(num_inputs: usize, layer_sizes: Vec<usize>) -> Self {
        let mut all_layer_sizes = vec![num_inputs];
        all_layer_sizes.extend(layer_sizes);

        let mut layers = Vec::new();
        for (input_size, output_size) in all_layer_sizes.iter().zip(all_layer_sizes.iter().skip(1))
        {
            layers.push(Layer::new(*input_size, *output_size));
        }
        Self { layers }
    }

    pub fn forward(&self, x: &Vec<MutableScalarTensor>) -> MutableScalarTensor {
        // TODO: remove the skip
        let first_layer_forward = self.layers[0].forward(x);
        let mut layer_result = match first_layer_forward {
            ForwardResult::Multi(intermediate_result) => intermediate_result,
            ForwardResult::Single(final_output) => return final_output,
        };

        for l in self.layers.iter().skip(1) {
            let result = match l.forward(&layer_result) {
                ForwardResult::Multi(intermediate_result) => intermediate_result,
                ForwardResult::Single(final_output) => return final_output,
            };
            layer_result = result;
        }

        panic!("No output was calculated! Are you sure you set the last layer to have a single neuron?");
    }
    
    pub fn scalar_forward(&self, x: &Vec<f32>) -> MutableScalarTensor {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_forward() {
        let n = Neuron::new(3);
        let out = n.forward(&vec![
            ScalarTensor::new(1.0),
            ScalarTensor::new(2.0),
            ScalarTensor::new(3.0),
        ]);
        let expected_result =
            &n.weights[0] * 1.0 + &n.weights[1] * 2.0 + &n.weights[2] * 3.0 + n.bias;
        {
            let out = out.borrow();
            assert_eq!(expected_result.borrow().data, out.data);
        }
    }

    #[test]
    fn test_neuron_parameters() {
        let n = Neuron::new(3);
        // 3 + 1 for the bias
        assert_eq!(4, n.parameters().len());
    }

    #[test]
    fn test_mlp() {
        let mlp = MLP::new(10, vec![3, 2, 1]);
        // 33 + 8 + 3
        assert_eq!(44, mlp.parameters().len());
    }
}
