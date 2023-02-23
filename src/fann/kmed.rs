use serde::{self, Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    iter::repeat,
};

use crate::{
    info::Info, BuildParams, Cache, Distance, DistanceCmp, EmbeddingProvider, LocalDistance, Tree,
};

const HIGHLIGHT_A: &str = "*";
const HIGHLIGHT_B: &str = ":";
const NO_HIGHLIGHT: &str = "";

#[derive(Serialize, Deserialize)]
struct Child {
    node: Node,
    center_dist: DistanceCmp,
}

#[derive(Serialize, Deserialize)]
struct Node {
    centroid_index: usize,
    radius: DistanceCmp,
    children: Vec<Child>,
}

impl Node {
    fn new(centroid_index: usize) -> Self {
        Node {
            centroid_index,
            radius: DistanceCmp::zero(),
            children: Vec::new(),
        }
    }

    fn is_before_leaf(&self) -> bool {
        self.children.iter().all(|c| c.node.children.is_empty())
    }

    fn get_dist<'a, E, D, T, I>(
        &self,
        ldist: &LocalDistance<'a, E, D, T>,
        info: &mut I,
    ) -> DistanceCmp
    where
        E: EmbeddingProvider<D, T>,
        D: Distance<T>,
        I: Info,
    {
        ldist.distance_cmp(self.centroid_index, info)
    }

    fn get_dist_min(&self, dist: &DistanceCmp) -> DistanceCmp {
        dist.combine(&self.radius, |d, radius| f64::max(0.0, d - radius))
    }

    fn get_child_dist_max(child: &Child) -> DistanceCmp {
        child
            .center_dist
            .combine(&child.node.radius, |center_dist, radius| {
                center_dist + radius
            })
    }

    fn compute_radius(&mut self) {
        self.radius = self
            .children
            .iter()
            .map(|child| Node::get_child_dist_max(child))
            .max()
            .unwrap_or(DistanceCmp::zero());
    }

    fn add_child<E, D, T, C, I>(&mut self, child: Node, provider: &E, cache: &mut C, info: &mut I)
    where
        E: EmbeddingProvider<D, T>,
        D: Distance<T>,
        C: Cache,
        I: Info,
    {
        let center_dist =
            provider.dist_internal(self.centroid_index, child.centroid_index, cache, info);
        self.children.push(Child {
            node: child,
            center_dist,
        });
        self.children
            .sort_unstable_by(|a, b| a.center_dist.cmp(&b.center_dist).reverse());
    }

    fn get_closest<'a, E, D, T, I>(
        &self,
        res: &mut Vec<(usize, DistanceCmp)>,
        own_dist: DistanceCmp,
        count: usize,
        ldist: &LocalDistance<'a, E, D, T>,
        info: &mut I,
    ) where
        E: EmbeddingProvider<D, T>,
        D: Distance<T>,
        I: Info,
    {
        fn max_dist(res: &Vec<(usize, DistanceCmp)>, count: usize) -> DistanceCmp {
            let index = count.min(res.len()) - 1;
            res[index].1
        }

        fn add_node(
            res: &mut Vec<(usize, DistanceCmp)>,
            node: &Node,
            distance: DistanceCmp,
            count: usize,
        ) {
            let element = (node.centroid_index, distance);
            let mindex = res.binary_search_by(|&(_, dist)| dist.cmp(&distance));
            match mindex {
                Ok(index) => res.insert(index, element),
                Err(index) => res.insert(index, element),
            }
            res.truncate(count);
        }

        if res.len() < count || own_dist < max_dist(res, count) {
            add_node(res, self, own_dist, count);
        }
        let is_outer = self.radius < own_dist;
        info.log_scan(self.centroid_index, is_outer);
        if is_outer {
            for child in self.children.iter() {
                let c_dist_est = own_dist.combine(&child.center_dist, |own, center| own - center);
                if max_dist(res, count) < c_dist_est {
                    continue;
                }
                let cdist = child.node.get_dist(ldist, info);
                child.node.get_closest(res, cdist, count, ldist, info);
            }
        } else {
            let mut inners: Vec<(&Node, DistanceCmp, DistanceCmp)> = self
                .children
                .iter()
                .map(|child| {
                    let cdist = child.node.get_dist(ldist, info);
                    let cmin = child.node.get_dist_min(&cdist);
                    (&child.node, cdist, cmin)
                })
                .collect();
            inners.sort_unstable_by(|(_, _, dist_a), (_, _, dist_b)| dist_a.cmp(dist_b));
            for (cnode, cdist, cmin) in inners.into_iter() {
                if max_dist(res, count) < cmin {
                    continue;
                }
                cnode.get_closest(res, cdist, count, ldist, info);
            }
        }
    }

    fn draw(
        &self,
        pad: usize,
        show_ixs: &HashMap<usize, bool>,
        stats: &HashMap<usize, &str>,
        prune: bool,
        radius: bool,
    ) -> Vec<String> {
        let highlight = match show_ixs.get(&self.centroid_index) {
            Some(true) => HIGHLIGHT_A,
            Some(false) => HIGHLIGHT_B,
            None => NO_HIGHLIGHT,
        };
        let num = format!(
            "{ix: >pad$}",
            ix = self.centroid_index,
            pad = pad - highlight.len(),
        );
        let rad = if radius {
            format!("[r:{r}]", r = self.radius.to())
        } else {
            "".to_owned()
        };
        let own = format!(
            "({highlight}{num}{rad})",
            highlight = highlight,
            num = num,
            rad = rad,
        );
        let children_count = self.children.len();
        if children_count == 0 {
            return Vec::from([own]);
        }
        if !radius && self.is_before_leaf() {
            let mut chs = self
                .children
                .iter()
                .map(|child| {
                    let cix = child.node.centroid_index;
                    let chighlight = match show_ixs.get(&cix) {
                        Some(true) => HIGHLIGHT_A,
                        Some(false) => HIGHLIGHT_B,
                        None => NO_HIGHLIGHT,
                    };
                    format!("{chighlight}{cix}", chighlight = chighlight, cix = cix)
                })
                .collect::<Vec<String>>()
                .join(", ");
            if prune && !chs.contains(&HIGHLIGHT_A) && !chs.contains(&HIGHLIGHT_B) {
                chs = "...".to_string();
            }
            return Vec::from([format!("{own}━({chs})", own = own, chs = chs)]);
        }
        let bar: String = repeat(" ").take(own.len()).collect();
        let sown = own.as_str();
        let sbar = bar.as_str();
        self.children
            .iter()
            .map(|child| {
                (
                    child.node.centroid_index,
                    child.node.draw(pad, &show_ixs, &stats, prune, radius),
                )
            })
            .enumerate()
            .map(|(cix, (child_ix, mut lines))| {
                let all_lines = lines.join("");
                if prune && !all_lines.contains(&HIGHLIGHT_A) && !all_lines.contains(&HIGHLIGHT_B) {
                    lines = Vec::from(["(...)".to_owned()]);
                }
                let cix = cix;
                let child_ix = child_ix;
                lines.into_iter().enumerate().map(move |(lix, line)| {
                    let start = if lix == 0 && cix == 0 { sown } else { sbar };
                    let mid: String = if lix == 0 {
                        let mid = if cix == 0 {
                            if children_count > 1 {
                                "┳"
                            } else {
                                "━"
                            }
                        } else {
                            if cix >= children_count - 1 {
                                "┗"
                            } else {
                                "┣"
                            }
                        };
                        match stats.get(&child_ix) {
                            Some(state) => state.to_uppercase().chars().nth(0).unwrap().to_string(),
                            None => mid.to_owned(),
                        }
                    } else {
                        if cix >= children_count - 1 {
                            " "
                        } else {
                            "┃"
                        }
                        .to_owned()
                    };
                    format!("{start}{mid}{line}", start = start, mid = mid, line = line)
                })
            })
            .flatten()
            .collect::<Vec<String>>()
    }
}

#[derive(Serialize, Deserialize)]
pub struct FannTree {
    root: Node,
    hash: String,
    distance_name: String,
}

impl FannTree {
    fn centroid<E, D, T, C, I>(
        provider: &E,
        all_ixs: &Vec<usize>,
        cache: &mut C,
        info: &mut I,
    ) -> usize
    where
        E: EmbeddingProvider<D, T>,
        D: Distance<T>,
        C: Cache,
        I: Info,
    {
        let (res_ix, _) =
            all_ixs
                .iter()
                .fold((None, DistanceCmp::of(f64::INFINITY)), |best, &ix| {
                    let (best_ix, best_dist) = best;
                    let cur_dist: DistanceCmp =
                        all_ixs.iter().fold(DistanceCmp::zero(), |res, &oix| {
                            if oix == ix || res > best_dist {
                                res
                            } else {
                                res.combine(
                                    &provider.dist_internal(ix, oix, cache, info),
                                    |cur, dist| cur + dist,
                                )
                            }
                        });
                    if best_ix.is_none() || cur_dist < best_dist {
                        (Some(ix), cur_dist)
                    } else {
                        best
                    }
                });
        res_ix.unwrap()
    }

    fn kmedoid<E, D, T, C, I>(
        provider: &E,
        all_ixs: Vec<usize>,
        init_centroids: Option<Vec<usize>>,
        k_num: usize,
        cache: &mut C,
        info: &mut I,
    ) -> Vec<(usize, Vec<usize>)>
    where
        E: EmbeddingProvider<D, T>,
        D: Distance<T>,
        C: Cache,
        I: Info,
    {
        if all_ixs.len() <= k_num {
            return all_ixs.iter().map(|&ix| (ix, Vec::new())).collect();
        }
        let buff_size = 10;
        let mut rounds = 1000;
        let mut buff: VecDeque<Vec<usize>> = VecDeque::with_capacity(buff_size);
        if let Some(init_centroids) = init_centroids {
            buff.push_front(init_centroids);
        } else {
            buff.push_front(all_ixs[..k_num].to_vec());
        }
        let mut done = false;
        loop {
            let centroids: Vec<usize> = buff.get(0).unwrap().clone();
            let mut res: Vec<(usize, Vec<usize>)> =
                centroids.iter().map(|&ix| (ix, Vec::from([ix]))).collect();
            all_ixs
                .iter()
                .filter(|&ix| !centroids.contains(ix))
                .for_each(|&ix| {
                    let (_, best) = res
                        .iter_mut()
                        .min_by(|(a, _), (b, _)| {
                            let dist_a = provider.dist_internal(ix, *a, cache, info);
                            let dist_b = provider.dist_internal(ix, *b, cache, info);
                            dist_a.cmp(&dist_b)
                        })
                        .unwrap();
                    best.push(ix);
                });
            if done {
                return res;
            }
            rounds -= 1;
            if rounds <= 0 {
                eprintln!("exhausted iteration steps");
                return res;
            }
            let new_cs: Vec<usize> = res
                .iter()
                .map(|(_, assignments)| Self::centroid(provider, assignments, cache, info))
                .collect();
            if buff.iter().any(|old_cs| *old_cs == new_cs) {
                done = true;
            }
            while buff.len() >= buff_size {
                buff.pop_back();
            }
            buff.push_front(new_cs.clone());
        }
    }

    fn remove(ixs: &mut Vec<usize>, index: usize) {
        ixs.retain(|&ix| ix != index);
    }

    fn build_level<E, D, T, C, I>(
        provider: &E,
        cache: &mut C,
        info: &mut I,
        cur_root_ix: usize,
        cur_all_ixs: Vec<usize>,
        max_node_size: usize,
    ) -> Node
    where
        E: EmbeddingProvider<D, T>,
        D: Distance<T>,
        C: Cache,
        I: Info,
    {
        let mut node = Node::new(cur_root_ix);
        let num_k = if max_node_size * max_node_size > cur_all_ixs.len() {
            ((cur_all_ixs.len() as f64).sqrt() as usize).max(1)
        } else {
            max_node_size
        };
        if num_k == 1 || cur_all_ixs.len() <= num_k {
            cur_all_ixs.iter().for_each(|&ix| {
                let mut cnode = Node::new(ix);
                cnode.compute_radius();
                node.add_child(cnode, provider, cache, info);
            });
        } else {
            let init_centroids = None;
            Self::kmedoid(provider, cur_all_ixs, init_centroids, num_k, cache, info)
                .into_iter()
                .for_each(|(centroid_ix, mut assignments)| {
                    Self::remove(&mut assignments, centroid_ix);
                    let child_node = Self::build_level(
                        provider,
                        cache,
                        info,
                        centroid_ix,
                        assignments,
                        max_node_size,
                    );
                    node.add_child(child_node, provider, cache, info);
                });
        }
        node.compute_radius();
        node
    }
}

pub struct FannBuildParams {
    pub max_node_size: Option<usize>,
}

impl BuildParams for FannBuildParams {}

impl<E, D, T> Tree<FannBuildParams, E, D, T> for FannTree
where
    E: EmbeddingProvider<D, T>,
    D: Distance<T>,
{
    fn build<C, I>(provider: &E, params: &FannBuildParams, cache: &mut C, info: &mut I) -> Self
    where
        C: Cache,
        I: Info,
    {
        let mut all_ixs: Vec<usize> = provider.all().collect();
        let max_node_size = match params.max_node_size {
            Some(max_node_size) => max_node_size,
            None => all_ixs.len(),
        };
        let root_ix = Self::centroid(provider, &all_ixs, cache, info);

        Self::remove(&mut all_ixs, root_ix);
        Self {
            root: Self::build_level(provider, cache, info, root_ix, all_ixs, max_node_size),
            hash: provider.compute_hash(),
            distance_name: provider.distance().name().to_string(),
        }
    }

    fn draw<I>(
        &self,
        high_ix: usize,
        info: Option<&I>,
        res: Option<Vec<(usize, f64)>>,
        prune: bool,
        radius: bool,
    ) -> String
    where
        I: Info,
    {
        let pad = format!("{high_ix}", high_ix = high_ix).len();
        let show_ixs: HashMap<usize, bool> = {
            let mut show_ixs = HashMap::new();
            if let Some(info) = info {
                info.dist_vec().iter().for_each(|ix| {
                    show_ixs.insert(*ix, false);
                });
            }
            if let Some(res) = res {
                res.iter().for_each(|(ix, _)| {
                    show_ixs.insert(*ix, true);
                });
            }
            show_ixs
        };
        let stats: HashMap<usize, &str> = {
            let mut stats = HashMap::new();
            if let Some(info) = info {
                stats.extend(info.scan_map());
            }
            stats
        };
        self.root
            .draw(pad, &show_ixs, &stats, prune, radius)
            .join("\n")
    }

    fn get_closest<'a, I>(
        &self,
        count: usize,
        ldist: &LocalDistance<'a, E, D, T>,
        info: &mut I,
    ) -> Vec<(usize, f64)>
    where
        I: Info,
    {
        let mut res: Vec<(usize, DistanceCmp)> = Vec::with_capacity(count + 1);
        let root_dist = self.root.get_dist(ldist, info);
        self.root
            .get_closest(&mut res, root_dist, count, ldist, info);
        res.iter()
            .map(|(ix, v)| (*ix, ldist.finalize_distance(v)))
            .collect()
    }

    fn fingerprint(&self) -> (&str, &str) {
        (&self.hash, &self.distance_name)
    }
}
