use std::{collections::HashSet, fmt::Display, vec};

pub struct ScalarTensorFactory {
    tensors: Vec<ScalarTensor>,
}

impl ScalarTensorFactory {
    pub fn new() -> Self {
        Self { tensors: vec![] }
    }

    pub fn tensor(&mut self, data: f32) -> usize {
        let index = self.tensors.len();
        self.tensors.push(ScalarTensor::new_leaf(data, index));
        index
    }

    pub fn addition_tensor(&mut self, lhs_index: usize, rhs_index: usize) -> usize {
        let index = self.tensors.len();
        let lhs = self.tensors[lhs_index].data;
        let rhs = self.tensors[rhs_index].data;
        self.tensors.push(ScalarTensor::new(
            lhs + rhs,
            index,
            vec![lhs_index, rhs_index], // maybe hashset?
            Op::ADD,
        ));
        index
    }

    fn build_topo(
        &self,
        current: &ScalarTensor,
        visited: &mut HashSet<usize>,
        topo: &mut Vec<usize>,
    ) {
        match visited.get(&current.index) {
            Some(_) => (),
            None => {
                visited.insert(current.index);
                for child in current.children.iter() {
                    self.build_topo(&self.tensors[*child], visited, topo);
                }
                topo.push(current.index);
            }
        }
    }

    fn build_reversed_topo(&self, tensor: &ScalarTensor) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut topo = Vec::new();
        self.build_topo(tensor, &mut visited, &mut topo);
        topo.reverse();
        topo
    }

    fn grad_fn(&mut self, index: usize) {
        match self.tensors[index].op {
            Op::NONE => (),
            Op::ADD => {
                // Apply the chain rule by getting the local derivative (which is 1 for addition) and multiplying
                // by the up until now calculated derivative
                let current_tensor = &self.tensors[index];
                let grad = current_tensor.grad;
                let indexes = current_tensor.children.clone();
                for child_index in indexes {
                    self.tensors[child_index].grad += grad * 1.0; // just for clarity
                }
            }
        }
    }

    pub fn backward(&mut self, index: usize) {
        self.tensors[index].grad = 1.0;
        let tensor = &self.tensors[index];
        let topo = self.build_reversed_topo(tensor);
        for index in topo {
            self.grad_fn(index);
        }
    }
}

///
/// Scalar tensor is a single value object that stores it's
/// data and it's gradient.
/// In order for backpropogation to work correctly we also need to store
/// the building blocks of this tensor - the so called children.
/// Childen is a usize vec because I don't know enough rust and I couldn't
/// get this to store references to the ScalarTensors that created this, so instead
/// we store an index into a singleton object that holds all ScalarTensors
/// Index is the index inside that singleton
pub struct ScalarTensor {
    pub data: f32,
    pub grad: f32,
    index: usize,
    children: Vec<usize>,
    op: Op,
}

enum Op {
    NONE,
    ADD,
}

impl Display for ScalarTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let op_string = match self.op {
            Op::NONE => "",
            Op::ADD => "+",
        };
        write!(
            f,
            "(data: {} | grad: {} | op: {})",
            self.data, self.grad, op_string
        )
    }
}

impl ScalarTensor {
    fn new_leaf(data: f32, index: usize) -> Self {
        Self {
            data,
            grad: 0.0,
            index,
            children: vec![],
            op: Op::NONE,
        }
    }

    fn new(data: f32, index: usize, children: Vec<usize>, op: Op) -> Self {
        Self {
            data,
            grad: 0.0,
            index,
            children,
            op,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_factory() {
        let mut f = ScalarTensorFactory::new();
        let index = f.tensor(5.0);
        let t = &f.tensors[index];
        assert_eq!(1, f.tensors.len());
        assert_eq!(5.0, t.data);
        assert_eq!(0, t.children.len());
        assert_eq!(0, t.index);

        let i2 = f.tensor(3.14);
        let t2 = &f.tensors[i2];
        assert_eq!(3.14, t2.data);
        assert_eq!(1, t2.index);
        assert_eq!(2, f.tensors.len());
    }

    #[test]
    fn test_addition_tensor() {
        let mut f = ScalarTensorFactory::new();
        let i1 = f.tensor(3.14);
        let i2 = f.tensor(5.0);
        let out_index = f.addition_tensor(i1, i2);
        let t = &f.tensors[out_index];
        assert_eq!(2, t.children.len());
        assert_eq!(8.14, t.data);
    }

    #[test]
    fn test_build_reversed_topo() {
        let mut f = ScalarTensorFactory::new();
        let i1 = f.tensor(3.14);
        let i2 = f.tensor(5.0);
        let out_index = f.addition_tensor(i1, i2);
        let topo = f.build_reversed_topo(&f.tensors[out_index]);
        assert_eq!(vec![out_index, i2, i1], topo);
    }

    #[test]
    fn test_addition_backward() {
        let mut f = ScalarTensorFactory::new();
        let i1 = f.tensor(3.14);
        let i2 = f.tensor(5.0);
        let out_index = f.addition_tensor(i1, i2);
        f.backward(out_index);
        assert_eq!(1.0, f.tensors[i1].grad);
        assert_eq!(1.0, f.tensors[i2].grad);
        assert_eq!(1.0, f.tensors[out_index].grad);
    }
}
