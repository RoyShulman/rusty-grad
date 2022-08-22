use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::Display,
    hash::{Hash, Hasher},
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

static OBJECT_COUNTER: AtomicUsize = AtomicUsize::new(0);

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
    unique_id: usize,
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

impl Hash for ScalarTensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.unique_id.hash(state);
    }
}

impl PartialEq for ScalarTensor {
    fn eq(&self, other: &Self) -> bool {
        self.unique_id == other.unique_id
    }
}

impl Eq for ScalarTensor {}

pub fn add(rhs: &MutableScalarTensor, lhs: &MutableScalarTensor) -> MutableScalarTensor {
    let new_tensor = Rc::new(RefCell::new(ScalarTensor {
        data: rhs.borrow().data + lhs.borrow().data,
        grad: 0.0,
        children: Vec::new(),
        op: Op::ADD,
        unique_id: OBJECT_COUNTER.fetch_add(1, Ordering::SeqCst),
    }));

    {
        let mut tmp = new_tensor.borrow_mut();
        tmp.children.push(rhs.clone());
        tmp.children.push(lhs.clone());
    }
    new_tensor
}

fn get_topo_graph_for_tensor(
    tensor: &MutableScalarTensor,
    visited: &mut HashSet<usize>,
    topo_graph: &mut Vec<MutableScalarTensor>,
) {
    if visited.contains(&tensor.borrow().unique_id) {
        return;
    }

    visited.insert(tensor.borrow().unique_id);
    for child in tensor.borrow().children.iter() {
        get_topo_graph_for_tensor(child, visited, topo_graph);
    }
    topo_graph.push(tensor.clone());
}

fn get_reversed_topo_graph(tensor: &MutableScalarTensor) -> Vec<MutableScalarTensor> {
    let mut topo_graph = Vec::new();
    get_topo_graph_for_tensor(tensor, &mut HashSet::new(), &mut topo_graph);
    topo_graph.reverse();
    topo_graph
}

// Apply back backpropagation to this tensor
pub fn backward(tensor: &MutableScalarTensor) {
    // First we initialize the root node with gradient of 1.0
    tensor.borrow_mut().grad = 1.0;
    let topo_graph = get_reversed_topo_graph(tensor);
    for t in topo_graph.iter() {
        // Now apply the grad function on each tensor
        t.borrow().grad_fn();
    }
}

impl ScalarTensor {
    pub fn new(data: f32) -> MutableScalarTensor {
        Rc::new(RefCell::new(Self {
            data,
            grad: 0.0,
            children: Vec::new(),
            op: Op::NONE,
            unique_id: OBJECT_COUNTER.fetch_add(1, Ordering::SeqCst),
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

    #[test]
    fn test_topo_graph() {
        let t1 = ScalarTensor::new(123.0);
        let t2 = ScalarTensor::new(11.0);
        let out1 = add(&t1, &t2);
        let out2 = add(&t1, &t2);
        let out3 = add(&out1, &out2);

        let topo_graph = get_reversed_topo_graph(&out3);
        let expected_vec = vec![
            out3.borrow(),
            out2.borrow(),
            out1.borrow(),
            t2.borrow(),
            t1.borrow(),
        ];

        assert_eq!(expected_vec.len(), topo_graph.len());
        for (expected, real) in expected_vec.iter().zip(topo_graph.iter()) {
            assert_eq!(expected.unique_id, real.borrow().unique_id);
        }
    }

    #[test]
    fn test_backward() {
        let t1 = ScalarTensor::new(123.0);
        let t2 = ScalarTensor::new(11.0);
        let out1 = add(&t1, &t2);
        let out2 = add(&t1, &t2);
        let out3 = add(&out1, &out2);

        backward(&out3);
        for t in vec![out1, out2, out3] {
            assert_eq!(1.0, t.borrow().grad);
        }

        // t1 and t2 are both contributing to 2 other tensors, so their gradient should
        // be 2.0. This can be shown as
        // out3 = out2 + out1 = (t1 + t2) + (t1 + t2)
        // Thus if we increase t1 by h for example we get
        // (t1 + h + t2) + (t1 +h + t2) = out3 + 2h
        // a change in t1 will have double the change on out3
        for t in vec![t1, t2] {
            assert_eq!(2.0, t.borrow().grad);
        }
    }
}
