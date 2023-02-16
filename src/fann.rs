use rayon::prelude::*;
use std::{
    collections::{HashMap, VecDeque},
    iter::repeat,
    marker::PhantomData,
};

const HIGHLIGHT_A: &str = "*";
const HIGHLIGHT_B: &str = ":";
const NO_HIGHLIGHT: &str = "";

#[derive(Debug, Clone, Copy)]
pub struct Embedding<T> {
    pub value: T,
    pub index: Option<usize>,
}

pub trait Distance<'a, T: 'a> {
    fn distance(&self, a: &Embedding<T>, b: &Embedding<T>) -> f64;
    fn get(&'a self, index: usize) -> T;
    fn all(&self) -> std::ops::Range<usize>;
}

#[derive(Hash, Eq, PartialEq, Debug)]
pub struct Key {
    lower_index: usize,
    upper_index: usize,
}

impl Key {
    pub fn new(index_a: usize, index_b: usize) -> Self {
        Key {
            lower_index: index_a.min(index_b),
            upper_index: index_a.max(index_b),
        }
    }
}

pub trait Cache {
    fn get(&mut self, key: &Key) -> Option<f64>;
    fn put(&mut self, key: Key, value: f64);
}

fn cached_dist<T, F, C>(a: &Embedding<T>, b: &Embedding<T>, cache: &mut C, dist: F) -> f64
where
    C: Cache,
    F: Fn(&Embedding<T>, &Embedding<T>) -> f64,
{
    match (a.index, b.index) {
        (None, _) => dist(a, b),
        (_, None) => dist(a, b),
        (Some(index_a), Some(index_b)) => {
            let key = Key::new(index_a, index_b);
            match cache.get(&key) {
                Some(res) => res,
                None => {
                    let res = dist(a, b);
                    cache.put(Key::new(index_a, index_b), res);
                    res
                }
            }
        }
    }
}

struct Child<'a, D, C, T: 'a>
where
    D: Distance<'a, T>,
    C: Cache,
{
    node: Node<'a, D, C, T>,
    center_dist: f64,
}

struct Node<'a, D, C, T: 'a>
where
    D: Distance<'a, T>,
    C: Cache,
{
    centroid_index: usize,
    radius: f64,
    count: usize,
    children: Vec<Child<'a, D, C, T>>,
    distance_type: PhantomData<&'a D>,
    embed_type: PhantomData<T>,
    cache_type: PhantomData<C>,
}

impl<'a, D, C, T: 'a> Node<'a, D, C, T>
where
    D: Distance<'a, T>,
    C: Cache,
{
    fn new(centroid_index: usize) -> Self {
        Node {
            centroid_index,
            radius: 0.0,
            count: 1,
            children: Vec::new(),
            distance_type: PhantomData,
            embed_type: PhantomData,
            cache_type: PhantomData,
        }
    }

    fn get_embed(&self, distance: &'a D) -> Embedding<T> {
        Embedding {
            value: distance.get(self.centroid_index),
            index: Some(self.centroid_index),
        }
    }

    fn count_descendants(&self) -> usize {
        self.count
    }

    fn count_children(&self) -> usize {
        self.children.len()
    }

    fn get_dist(&self, embed: &Embedding<T>, distance: &'a D, cache: &mut C) -> f64 {
        cached_dist(&self.get_embed(distance), embed, cache, |a, b| {
            distance.distance(a, b)
        })
    }

    fn get_dist_min(&self, embed: &Embedding<T>, distance: &'a D, cache: &mut C) -> f64 {
        f64::max(0.0, self.get_dist(embed, distance, cache) - self.radius)
    }

    fn get_child_dist_max(child: &Child<'a, D, C, T>) -> f64 {
        child.center_dist + child.node.radius
    }

    fn compute_radius(&mut self) {
        self.radius = self
            .children
            .iter()
            .map(|child| Node::get_child_dist_max(child))
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
    }

    fn add_child(&mut self, child: Node<'a, D, C, T>, distance: &'a D, cache: &mut C) {
        let center_dist = self.get_dist(&child.get_embed(distance), distance, cache);
        self.count += child.count_descendants();
        self.children.push(Child {
            node: child,
            center_dist,
        });
        self.children
            .sort_unstable_by(|a, b| a.center_dist.partial_cmp(&b.center_dist).unwrap().reverse());
    }

    fn get_closest<'t>(
        &'t self,
        res: &mut Vec<(usize, f64)>,
        embed: &Embedding<T>,
        count: usize,
        distance: &'a D,
        cache: &mut C,
    ) {
        let own_dist = self.get_dist(embed, distance, cache);

        fn max_dist(res: &Vec<(usize, f64)>, count: usize) -> f64 {
            let index = count.min(res.len()) - 1;
            res.get(index).unwrap().1
        }

        fn add_node<'a, D, C, T: 'a>(
            res: &mut Vec<(usize, f64)>,
            node: &Node<'a, D, C, T>,
            distance: f64,
            count: usize,
        ) where
            D: Distance<'a, T>,
            C: Cache,
        {
            let element = (node.centroid_index, distance);
            let mindex = res.binary_search_by(|&(_, dist)| dist.partial_cmp(&distance).unwrap());
            match mindex {
                Ok(index) => res.insert(index, element),
                Err(index) => res.insert(index, element),
            }
            res.truncate(count);
        }

        if res.len() < count || own_dist < max_dist(res, count) {
            add_node(res, self, own_dist, count);
        }
        if self.radius < own_dist {
            for child in self.children.iter() {
                if max_dist(res, count) < own_dist - child.center_dist {
                    continue;
                }
                child.node.get_closest(res, embed, count, distance, cache);
            }
        } else {
            let mut inners: Vec<(&Node<D, C, T>, f64)> = self
                .children
                .iter()
                .map(|child| (&child.node, child.node.get_dist_min(embed, distance, cache)))
                .collect::<Vec<(&Node<D, C, T>, f64)>>();
            inners.sort_unstable_by(|(_, dist_a), (_, dist_b)| dist_a.partial_cmp(dist_b).unwrap());
            for (cnode, cmin) in inners.iter() {
                if max_dist(res, count) < *cmin {
                    continue;
                }
                cnode.get_closest(res, embed, count, distance, cache);
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
            format!("[r:{r}]", r = self.radius)
        } else {
            "".to_owned()
        };
        let own = format!(
            "({highlight}{num}{rad})",
            highlight = highlight,
            num = num,
            rad = rad,
        );
        let children_count = self.count_children();
        if children_count == 0 {
            return Vec::from([own]);
        }
        if !radius && self.count_descendants() == children_count + 1 {
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
        let bar = repeat(" ").take(own.len()).collect::<String>();
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

pub struct Fann<'a, D, C, T: 'a>
where
    D: Distance<'a, T>,
    C: Cache,
{
    distance: &'a D,
    cache: &'a mut C,
    root: Option<Node<'a, D, C, T>>,
    high_ix: usize,
}

impl<'a, D, C, T: 'a> Fann<'a, D, C, T>
where
    D: Distance<'a, T>,
    C: Cache,
{
    pub fn new(distance: &'a D, cache: &'a mut C) -> Self {
        Fann {
            distance,
            cache,
            root: None,
            high_ix: 0,
        }
    }

    pub fn as_embedding(&self, embed: T) -> Embedding<T> {
        Embedding {
            value: embed,
            index: None,
        }
    }

    fn get_embed(&self, index: usize) -> Embedding<T> {
        Embedding {
            value: self.distance.get(index),
            index: Some(index),
        }
    }

    fn get_dist(&mut self, embed_a: &Embedding<T>, embed_b: &Embedding<T>) -> f64 {
        cached_dist(embed_a, embed_b, self.cache, |a, b| {
            self.distance.distance(a, b)
        })
    }

    fn centroid(&mut self, all_ixs: &Vec<usize>) -> usize {
        let (res_ix, _) = all_ixs.iter().fold((None, f64::INFINITY), |best, &ix| {
            let (best_ix, best_dist) = best;
            let embed = self.get_embed(ix);
            let cur_dist: f64 = all_ixs.iter().fold(0.0, |res, &oix| {
                if oix == ix || res > best_dist {
                    res
                } else {
                    let oembed = self.get_embed(oix);
                    res + self.get_dist(&embed, &oembed)
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

    fn kmedoid(&mut self, all_ixs: Vec<usize>, k_num: usize) -> Vec<(usize, Vec<usize>)> {
        if all_ixs.len() <= k_num {
            return all_ixs.iter().map(|&ix| (ix, Vec::new())).collect();
        }
        let buff_size = 10;
        let mut rounds = 1000;
        let mut buff: VecDeque<Vec<usize>> = VecDeque::with_capacity(buff_size);
        buff.push_front(all_ixs[..k_num].to_vec());
        let mut done = false;
        loop {
            let centroids: Vec<usize> = buff.get(0).unwrap().clone();
            let mut res: Vec<(usize, Vec<usize>)> =
                centroids.iter().map(|&ix| (ix, Vec::from([ix]))).collect();
            all_ixs
                .iter()
                .filter(|&ix| !centroids.contains(ix))
                .for_each(|&ix| {
                    let embed = self.get_embed(ix);
                    let (_, best) = res
                        .iter_mut()
                        .min_by(|(a, _), (b, _)| {
                            let dist_a = self.get_dist(&embed, &self.get_embed(*a));
                            let dist_b = self.get_dist(&embed, &self.get_embed(*b));
                            dist_a.partial_cmp(&dist_b).unwrap()
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
                .map(|(_, assignments)| self.centroid(assignments))
                .collect();
            if buff.par_iter().any(|old_cs| *old_cs == new_cs) {
                // TODO use par for actually useful things
                done = true;
            }
            while buff.len() >= buff_size {
                buff.pop_back();
            }
            buff.push_front(new_cs.clone());
        }
    }

    pub fn build(&mut self, max_node_size: Option<usize>) {
        let mut all_ixs: Vec<usize> = self.distance.all().collect();
        let max_node_size = match max_node_size {
            Some(max_node_size) => max_node_size,
            None => all_ixs.len(),
        };
        self.high_ix = *all_ixs.iter().max().unwrap();
        let root_ix = self.centroid(&all_ixs);

        fn remove(ixs: &mut Vec<usize>, index: usize) {
            ixs.retain(|&ix| ix != index);
        }

        fn build_level<'a, D: Distance<'a, T>, C: Cache, T>(
            tree: &mut Fann<'a, D, C, T>,
            cur_root_ix: usize,
            cur_all_ixs: Vec<usize>,
            max_node_size: usize,
        ) -> Node<'a, D, C, T> {
            let mut node: Node<'a, D, C, T> = Node::new(cur_root_ix);
            let num_k = if max_node_size * max_node_size > cur_all_ixs.len() {
                ((cur_all_ixs.len() as f64).sqrt() as usize).max(1)
            } else {
                max_node_size
            };
            if num_k == 1 || cur_all_ixs.len() <= num_k {
                cur_all_ixs.iter().for_each(|&ix| {
                    let mut cnode: Node<'a, D, C, T> = Node::new(ix);
                    cnode.compute_radius();
                    node.add_child(cnode, tree.distance, tree.cache);
                });
            } else {
                tree.kmedoid(cur_all_ixs, num_k).into_iter().for_each(
                    |(centroid_ix, mut assignments)| {
                        remove(&mut assignments, centroid_ix);
                        let child_node = build_level(tree, centroid_ix, assignments, max_node_size);
                        node.add_child(child_node, tree.distance, tree.cache);
                    },
                );
            }
            node.compute_radius();
            node
        }

        remove(&mut all_ixs, root_ix);
        self.root = Some(build_level(self, root_ix, all_ixs, max_node_size));
    }

    pub fn get_closest(&mut self, embed: &Embedding<T>, count: usize) -> Vec<(usize, f64)> {
        let mut res: Vec<(usize, f64)> = Vec::with_capacity(count + 1);
        self.root
            .as_ref()
            .unwrap()
            .get_closest(&mut res, embed, count, self.distance, self.cache);
        res
    }

    pub fn draw(&self, prune: bool, radius: bool) -> String {
        let pad = format!("{high_ix}", high_ix = self.high_ix).len();
        let show_ixs: HashMap<usize, bool> = HashMap::new();
        let stats: HashMap<usize, &str> = HashMap::new();
        self.root
            .as_ref()
            .unwrap()
            .draw(pad, &show_ixs, &stats, prune, radius)
            .join("\n")
    }
}
