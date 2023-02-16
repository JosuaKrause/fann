use ndarray::{ArrayView1, ArrayView2};

use crate::{Distance, Embedding};

pub struct DotDistance<'a> {
    arr: ArrayView2<'a, f64>,
}

impl<'a> DotDistance<'a> {
    pub fn new(arr: ArrayView2<'a, f64>) -> Self {
        DotDistance { arr }
    }
}

impl<'a> Distance<'a, ArrayView1<'a, f64>> for DotDistance<'a> {
    fn distance(
        &self,
        a: &Embedding<ArrayView1<'a, f64>>,
        b: &Embedding<ArrayView1<'a, f64>>,
    ) -> f64 {
        let res = a.value.dot(&b.value);
        (-res).exp()
    }

    fn get(&'a self, index: usize) -> ArrayView1<'a, f64> {
        self.arr.row(index)
    }

    fn all(&self) -> std::ops::Range<usize> {
        0..self.arr.shape()[0]
    }
}
