use std::{cell::RefCell, fmt::Display, rc::Rc};

///
/// Scalar tensor is a single value object that stores it's
/// data and it's gradient.
/// In order for backpropogation to work correctly we also need to store
/// the building blocks of this tensor - the so called children.
#[derive(Debug)]
pub struct ScalarTensor {
    pub data: f32,
    pub grad: f32,
    children: Vec<MutableScalarTensor>,
    op: Op,
}

type MutableScalarTensor = Rc<RefCell<ScalarTensor>>;

#[derive(Debug, PartialEq)]
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

pub fn add(rhs: &MutableScalarTensor, lhs: &MutableScalarTensor) -> MutableScalarTensor {
    let new_tensor = Rc::new(RefCell::new(ScalarTensor {
        data: rhs.borrow().data + lhs.borrow().data,
        grad: 0.0,
        children: Vec::new(),
        op: Op::ADD,
    }));

    {
        let mut tmp = new_tensor.borrow_mut();
        tmp.children.push(rhs.clone());
        tmp.children.push(lhs.clone());
    }
    new_tensor
}

impl ScalarTensor {
    pub fn new(data: f32) -> MutableScalarTensor {
        Rc::new(RefCell::new(Self {
            data,
            grad: 0.0,
            children: Vec::new(),
            op: Op::NONE,
        }))
    }

    fn grad_fn(&self) {
        match self.op {
            Op::NONE => (),
            Op::ADD => {
                for child in self.children.iter() {
                    // Local derivative of an addition is 1, and we apply the chain rule by
                    // multiplying by the total grad so far
                    child.borrow_mut().grad += self.grad * 1.0;
                }
            }
        }
    }

    // fn get_reversed_topo_graph(&self) ->

    // Apply back backpropagation to this tensor
    pub fn backward(&mut self) {
        self.grad = 1.0;
        self.grad_fn();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor() {
        let a = ScalarTensor::new(5.0);
        let t = a.borrow();
        assert_eq!(5.0, t.data);
        assert_eq!(0, t.children.len());
        assert_eq!(Op::NONE, t.op);
    }

    #[test]
    fn test_addition() {
        let t1 = ScalarTensor::new(5.0);
        let t2 = ScalarTensor::new(3.14);
        let out_tensor = add(&t1, &t2);
        let out = out_tensor.borrow();
        assert_eq!(8.14, out.data);
        assert_eq!(2, out.children.len());
        assert_eq!(Op::ADD, out.op);
    }

    #[test]
    fn test_grad_fn() {
        let t1 = ScalarTensor::new(5.0);
        let t2 = ScalarTensor::new(3.14);
        let out = add(&t1, &t2);
        assert_eq!(0.0, t1.borrow().grad);
        assert_eq!(0.0, t2.borrow().grad);
        out.borrow_mut().grad = 1.0;
        out.borrow().grad_fn();
        assert_eq!(1.0, t1.borrow().grad);
        assert_eq!(1.0, t2.borrow().grad);
    }
}
