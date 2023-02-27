use std::collections::BinaryHeap;

use crate::{info::Info, Distance, DistanceCmp, EmbeddingProvider, LocalDistance};

pub trait StreamingNode {
    fn get_index(&self) -> usize;

    fn get_distance<'a, E, D, T, I>(
        &self,
        ldist: &LocalDistance<'a, E, D, T>,
        info: &mut I,
    ) -> DistanceCmp
    where
        E: EmbeddingProvider<D, T>,
        D: Distance<T>,
        I: Info,
    {
        ldist.distance_cmp(self.get_index(), info)
    }

    fn get_radius(&self) -> DistanceCmp;

    fn with_children<'a, F, I>(
        &'a self,
        apply: F,
        queue: &mut BinaryHeap<StreamingElement<'a, Self>>,
        res: &mut Vec<(usize, DistanceCmp)>,
        info: &mut I,
    ) where
        F: Fn(
            &'a Self,
            &DistanceCmp,
            &mut Vec<(usize, DistanceCmp)>,
            &mut I,
        ) -> Option<StreamingElement<'a, Self>>,
        I: Info,
        Self: Sized + 'a;

    fn get_min_distance(&self, dist_cmp: &DistanceCmp) -> DistanceCmp;
}

pub enum MaybeDistance {
    Dist(DistanceCmp),
    DistMinEst(DistanceCmp),
}

pub struct StreamingElement<'a, R>
where
    R: StreamingNode,
{
    elem: &'a R,
    dist: MaybeDistance,
}

impl<'a, R> StreamingElement<'a, R>
where
    R: StreamingNode,
{
    pub fn new<E, D, T, I>(elem: &'a R, ldist: &LocalDistance<'a, E, D, T>, info: &mut I) -> Self
    where
        E: EmbeddingProvider<D, T>,
        D: Distance<T>,
        I: Info,
    {
        let dist = elem.get_distance(ldist, info);
        Self {
            elem,
            dist: MaybeDistance::Dist(dist),
        }
    }

    pub fn with_estimate(elem: &'a R, estimate: DistanceCmp) -> Self {
        Self {
            elem,
            dist: MaybeDistance::DistMinEst(estimate),
        }
    }

    pub fn get_index(&self) -> usize {
        self.elem.get_index()
    }

    pub fn get_radius(&self) -> DistanceCmp {
        self.elem.get_radius()
    }

    pub fn with_children<F, I>(
        &self,
        apply: F,
        queue: &mut BinaryHeap<StreamingElement<'a, R>>,
        res: &mut Vec<(usize, DistanceCmp)>,
        info: &mut I,
    ) where
        F: Fn(
            &'a R,
            &DistanceCmp,
            &mut Vec<(usize, DistanceCmp)>,
            &mut I,
        ) -> Option<StreamingElement<'a, R>>,
        I: Info,
    {
        self.elem.with_children(apply, queue, res, info)
    }

    fn get_cached_dist(&self) -> DistanceCmp {
        match self.dist {
            MaybeDistance::Dist(dist) => dist,
            MaybeDistance::DistMinEst(_) => {
                panic!("distance should have been computed at this point")
            }
        }
    }

    fn get_distance<E, D, T, I>(
        &mut self,
        ldist: &LocalDistance<'a, E, D, T>,
        info: &mut I,
    ) -> DistanceCmp
    where
        E: EmbeddingProvider<D, T>,
        D: Distance<T>,
        I: Info,
    {
        match self.dist {
            MaybeDistance::Dist(dist) => dist,
            MaybeDistance::DistMinEst(_) => {
                let res = self.elem.get_distance(ldist, info);
                self.dist = MaybeDistance::Dist(res);
                res
            }
        }
    }

    fn dist_min(&self) -> DistanceCmp {
        match self.dist {
            MaybeDistance::Dist(dist) => self.elem.get_min_distance(&dist),
            MaybeDistance::DistMinEst(estimate) => estimate,
        }
    }
}

impl<'a, R> Ord for StreamingElement<'a, R>
where
    R: StreamingNode,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist_min().cmp(&other.dist_min()).reverse()
    }
}

impl<'a, R> PartialOrd for StreamingElement<'a, R>
where
    R: StreamingNode,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a, R> Eq for StreamingElement<'a, R> where R: StreamingNode {}

impl<'a, R> PartialEq for StreamingElement<'a, R>
where
    R: StreamingNode,
{
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

pub trait StreamingNeighbors<E, D, T, R>
where
    E: EmbeddingProvider<D, T>,
    D: Distance<T>,
    R: StreamingNode,
{
    fn get_roots<'a, I>(
        &'a self,
        ldist: &LocalDistance<'a, E, D, T>,
        info: &mut I,
    ) -> Vec<StreamingElement<'a, R>>
    where
        I: Info;

    fn create_local_distance<'a>(&'a self, other: &'a T) -> LocalDistance<'a, E, D, T>;

    fn compute_closest<'a, I>(
        roots: Vec<StreamingElement<'a, R>>,
        ldist: &LocalDistance<'a, E, D, T>,
        count: usize,
        info: &mut I,
    ) -> Vec<(usize, f64)>
    where
        I: Info,
    {
        fn max_dist(res: &Vec<(usize, DistanceCmp)>, count: usize) -> DistanceCmp {
            if res.len() < count {
                return DistanceCmp::inf();
            }
            res[res.len() - 1].1
        }

        fn add_node<'a, R>(
            res: &mut Vec<(usize, DistanceCmp)>,
            elem: &StreamingElement<'a, R>,
            count: usize,
        ) where
            R: StreamingNode,
        {
            let dist = elem.get_cached_dist();
            let item = (elem.get_index(), dist);
            let mindex = res.binary_search_by_key(&dist, |(_, v)| *v);
            match mindex {
                Ok(index) => res.insert(index, item),
                Err(index) => res.insert(index, item),
            }
            res.truncate(count);
        }

        let mut res: Vec<(usize, DistanceCmp)> = Vec::with_capacity(count + 1);
        // roots
        //     .iter()
        //     .for_each(|root| add_node(&mut res, root, count));
        let mut queue: BinaryHeap<StreamingElement<'a, R>> = BinaryHeap::from(roots);
        while let Some(mut cur) = queue.pop() {
            if cur.dist_min() > max_dist(&res, count) {
                break;
            }
            cur.get_distance(ldist, info);
            let cur = cur;
            if cur.get_cached_dist() < max_dist(&res, count) {
                add_node(&mut res, &cur, count);
            }
            let own_dist = cur.get_cached_dist();
            let is_outer = cur.get_radius() < own_dist;
            info.log_scan(cur.get_index(), is_outer);
            if is_outer {
                cur.with_children(
                    |child, &center_dist, res, info| {
                        let c_dist_est = own_dist.combine(&center_dist, |own, center| own - center);
                        if max_dist(res, count) < c_dist_est {
                            return None;
                        }
                        let celem = StreamingElement::with_estimate(child, c_dist_est);
                        // let celem = StreamingElement::new(child, ldist, info);
                        // if max_dist(res, count) < celem.dist_min() {
                        //     return None;
                        // }
                        // add_node(res, &celem, count);
                        Some(celem)
                    },
                    &mut queue,
                    &mut res,
                    info,
                );
            } else {
                cur.with_children(
                    |child, _, res, info| {
                        let celem = StreamingElement::new(child, ldist, info);
                        if max_dist(res, count) < celem.dist_min() {
                            return None;
                        }
                        // add_node(res, &celem, count);
                        Some(celem)
                    },
                    &mut queue,
                    &mut res,
                    info,
                );
            }
        }
        res.into_iter()
            .map(|(ix, dist)| (ix, ldist.finalize_distance(&dist)))
            .collect()
    }

    fn get_closest_stream<I>(&self, other: &T, count: usize, info: &mut I) -> Vec<(usize, f64)>
    where
        I: Info,
    {
        let ldist = self.create_local_distance(other);
        let roots = self.get_roots(&ldist, info);
        Self::compute_closest(roots, &ldist, count, info)
    }
}
