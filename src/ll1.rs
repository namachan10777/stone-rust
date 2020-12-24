use std::collections::{HashMap, HashSet};
use std::fmt;

/// Caution! non-terminal symbol and terminal symbol have a uniq cardinal independently.
/// e.g. Term1 -> 0, Term2 -> 1, Term3 -> 2, Rule1 -> 0, Rule2 -> 1...
pub trait Terminal {
    fn cardinal(&self) -> usize;
    fn accept(&self) -> bool;
    const N: usize;

    fn id(&self) -> SymbolId {
        SymbolId::Term(self.cardinal())
    }
}

pub trait NonTerminal {
    fn cardinal(&self) -> usize;
    const N: usize;

    fn id(&self) -> SymbolId {
        SymbolId::NTerm(self.cardinal())
    }
}

#[derive(Debug, PartialEq, Hash, Eq, Clone)]
pub enum SymbolId {
    Term(usize),
    NTerm(usize),
}

use std::cmp::Ordering;

impl PartialOrd<Self> for SymbolId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (SymbolId::Term(_), SymbolId::NTerm(_)) => Some(Ordering::Less),
            (SymbolId::NTerm(_), SymbolId::Term(_)) => Some(Ordering::Greater),
            (SymbolId::Term(a), SymbolId::Term(b)) => a.partial_cmp(&b),
            (SymbolId::NTerm(a), SymbolId::NTerm(b)) => a.partial_cmp(&b),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum Error {
    SyntaxError,
    RuleMustBeNonTerminal,
}

pub enum ReduceSymbol<T, Ast> {
    Term(T),
    Ast(Ast),
}

pub type Words = Vec<Vec<SymbolId>>;
type Reducer<T, Ast> = Box<dyn FnOnce(&mut Vec<ReduceSymbol<T, Ast>>)>;
pub struct Rule<T, Ast> {
    words: Words,
    reducer: Reducer<T, Ast>,
}

impl<T, Ast> fmt::Debug for Rule<T, Ast> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        f.debug_list().entries(self.words.iter()).finish()
    }
}

impl<T, Ast> PartialEq<Rule<T, Ast>> for Rule<T, Ast> {
    fn eq(&self, other: &Self) -> bool {
        self.words.eq(&other.words)
    }
}

// key is a cardinal of non-terminal symbol
pub type Rules<T, Ast> = HashMap<SymbolId, Rule<T, Ast>>;
pub type IRules<T, Ast> = Vec<Rule<T, Ast>>;

fn gen_fluxed_and_lasts_rule(
    rules: &[Vec<SymbolId>],
) -> (Vec<SymbolId>, Option<Vec<Vec<SymbolId>>>) {
    let mut until_common_idx = 0;
    'linear_check: loop {
        let mut sample = rules[0].get(until_common_idx);
        for rule in rules {
            if rule.len() == until_common_idx || sample != rule.get(until_common_idx) {
                break 'linear_check;
            }
        }
        until_common_idx += 1;
        sample = rules[0].get(until_common_idx);
    }
    let common = rules[0][0..until_common_idx].to_vec();
    let tails = rules.iter().map(|rule| rule[until_common_idx..].to_owned());
    if tails.clone().all(|rule| rule.len() == 0) {
        (common, None)
    } else {
        (common, Some(tails.collect::<Vec<Vec<SymbolId>>>()))
    }
}

// internal impl
// 先頭が共通していれば共通部分単体のルールに書き換え、共通以後を新ルールにして追加、新ルールにも適用
fn remove_common_impl<'a, T, Ast>(rules: &mut IRules<T, Ast>, target_idx: usize) {
    rules[target_idx].words.sort_by(|rule1, rule2| {
        if rule1.is_empty() {
            Ordering::Less
        } else if rule2.is_empty() {
            Ordering::Greater
        } else {
            rule1[0].partial_cmp(&rule2[0]).unwrap()
        }
    });
    let mut begin = 0;
    let mut common_removed_rule = Vec::new();
    for i in 0..rules[target_idx].words.len() {
        if rules[target_idx].words[begin].get(0) != rules[target_idx].words[i].get(0) {
            if i - begin > 1 {
                let (mut replaced, new_rule) =
                    gen_fluxed_and_lasts_rule(&rules[target_idx].words[begin..i]);
                if let Some(new_rule) = new_rule {
                    replaced.push(SymbolId::NTerm(rules.len()));
                    rules.push(Rule {
                        words: new_rule,
                        reducer: Box::new(|_: &mut Vec<ReduceSymbol<T, Ast>>| {}),
                    });
                    remove_common_impl(rules, rules.len() - 1);
                }
                common_removed_rule.push(replaced);
            } else {
                common_removed_rule.push(rules[target_idx].words[begin].clone());
            }
            begin = i;
        }
    }
    if rules[target_idx].words.len() - begin > 1 {
        let (mut replaced, new_rule) = gen_fluxed_and_lasts_rule(
            &rules[target_idx].words[begin..rules[target_idx].words.len()],
        );
        if let Some(new_rule) = new_rule {
            replaced.push(SymbolId::NTerm(rules.len()));
            rules.push(Rule {
                words: new_rule,
                reducer: Box::new(|_: &mut Vec<ReduceSymbol<T, Ast>>| {}),
            });
            remove_common_impl(rules, rules.len() - 1);
        }
        common_removed_rule.push(replaced);
    } else {
        common_removed_rule.push(rules[target_idx].words[begin].clone());
    }
    rules[target_idx].words = common_removed_rule;
}

// 共通部分削除
// A B C D | A B C E | A B F
// -> A B -> (C (D | E) | F)
fn remove_common<T, Ast>(rules: Rules<T, Ast>) -> Result<IRules<T, Ast>, Error>
where
    //F: FnMut(&mut Vec<ReduceSymbol<T, Ast>>),
{
    let mut pairs = rules
        .into_iter()
        .map(|(id, rule)| {
            if let SymbolId::NTerm(id) = id {
                Ok((id, rule))
            } else {
                Err(Error::RuleMustBeNonTerminal)
            }
        })
        .collect::<Result<Vec<(usize, Rule<T, Ast>)>, Error>>()?;
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    let mut tbl = pairs
        .into_iter()
        .map(|(_, rule)| rule)
        .collect::<Vec<Rule<T, Ast>>>();
    for i in 0..tbl.len() {
        remove_common_impl(&mut tbl, i);
    }
    Ok(tbl)
}

#[cfg(test)]
mod test {
    use super::*;

    macro_rules! map {
        { $($key:expr => $value:expr),* } => {
            {
                let mut hash = HashMap::new();
                $(
                    hash.insert($key, $value);
                )*
                hash
            }
        };
    }

    macro_rules! set {
        { $($value:expr),* } => {
            {
                let mut hash = HashSet::new();
                $(
                    hash.insert($value);
                )*
                hash
            }
        };
    }

    #[derive(Debug, PartialEq)]
    enum Term {
        Num(usize),
        LP,
        RP,
        Add,
        Mul,
    }

    impl Terminal for Term {
        fn cardinal(&self) -> usize {
            match self {
                Term::Num(_) => 0,
                Term::LP => 1,
                Term::RP => 2,
                Term::Add => 3,
                Term::Mul => 4,
            }
        }

        const N: usize = 5;
        fn accept(&self) -> bool {
            if let Term::Num(_) = self {
                true
            } else {
                false
            }
        }
    }

    #[derive(Debug, PartialEq)]
    enum NTerm {
        Factor,
        Expr,
        Term,
    }

    impl NonTerminal for NTerm {
        const N: usize = 3;
        fn cardinal(&self) -> usize {
            match self {
                NTerm::Expr => 0,
                NTerm::Term => 1,
                NTerm::Factor => 2,
            }
        }
    }

    #[test]
    fn test_gen_fluxed_and_lasts_rule() {
        let (common, tails) = gen_fluxed_and_lasts_rule(&[
            vec![
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
            ],
            vec![
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(1),
                SymbolId::NTerm(0),
            ],
            vec![
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(1),
                SymbolId::NTerm(2),
            ],
        ]);
        assert_eq!(common, vec![SymbolId::NTerm(0), SymbolId::NTerm(0)]);
        assert_eq!(
            tails,
            Some(vec![
                vec![SymbolId::NTerm(0), SymbolId::NTerm(0)],
                vec![SymbolId::NTerm(1), SymbolId::NTerm(0)],
                vec![SymbolId::NTerm(1), SymbolId::NTerm(2)],
            ])
        );
        let (common, tails) = gen_fluxed_and_lasts_rule(&[
            vec![
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
            ],
            vec![
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
            ],
            vec![
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
            ],
        ]);
        assert_eq!(
            common,
            vec![
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(0)
            ]
        );
        assert_eq!(tails, None);
    }

    #[test]
    fn test_remove_common() {
        let reducer = |_: &mut Vec<ReduceSymbol<usize, usize>>| {};
        let rules = map! {
            NTerm::Expr.id()=>
                Rule::<usize, usize> {
                    words: vec![
                        vec![NTerm::Term.id()],
                        vec![NTerm::Term.id(), Term::Add.id(), NTerm::Expr.id()],
                    ],
                    reducer: Box::new(reducer.clone()),
                },
            NTerm::Term.id()=>
                Rule {
                    words: vec![
                        vec![NTerm::Factor.id()],
                        vec![NTerm::Factor.id(), Term::Mul.id(), NTerm::Term.id()],
                    ],
                    reducer: Box::new(reducer.clone()),
                },
            NTerm::Factor.id()=>
                Rule {
                    words: vec![
                        vec![Term::Num(0).id()],
                        vec![Term::LP.id(), NTerm::Expr.id(), Term::RP.id()],
                    ],
                    reducer: Box::new(reducer.clone()),
                }
        };
        let removed = vec![
            (Rule {
                words: vec![vec![NTerm::Term.id(), SymbolId::NTerm(3)]],
                reducer: Box::new(reducer.clone()),
            }),
            (Rule {
                words: vec![vec![NTerm::Factor.id(), SymbolId::NTerm(4)]],
                reducer: Box::new(reducer.clone()),
            }),
            (Rule {
                words: vec![
                    vec![Term::Num(0).id()],
                    vec![Term::LP.id(), NTerm::Expr.id(), Term::RP.id()],
                ],
                reducer: Box::new(reducer.clone()),
            }),
            (Rule {
                words: vec![vec![], vec![Term::Add.id(), NTerm::Expr.id()]],
                reducer: Box::new(reducer.clone()),
            }),
            (Rule {
                words: vec![vec![], vec![Term::Mul.id(), NTerm::Term.id()]],
                reducer: Box::new(reducer.clone()),
            }),
        ];
        assert_eq!(Ok(removed), remove_common(rules));
        let mut rules = HashMap::new();
        rules.insert(
            SymbolId::NTerm(0),
            Rule {
                words: vec![
                    vec![SymbolId::Term(1), SymbolId::Term(2), SymbolId::Term(3)],
                    vec![SymbolId::Term(1), SymbolId::Term(4), SymbolId::Term(3)],
                    vec![SymbolId::Term(1), SymbolId::Term(4), SymbolId::Term(5)],
                    vec![SymbolId::Term(1), SymbolId::Term(4), SymbolId::Term(5)],
                    vec![SymbolId::Term(0), SymbolId::Term(2), SymbolId::Term(3)],
                ],
                reducer: Box::new(reducer.clone()),
            },
        );
        let removed = vec![
            Rule {
                words: vec![
                    vec![SymbolId::Term(0), SymbolId::Term(2), SymbolId::Term(3)],
                    vec![SymbolId::Term(1), SymbolId::NTerm(1)],
                ],
                reducer: Box::new(reducer.clone()),
            },
            Rule {
                words: vec![
                    vec![SymbolId::Term(2), SymbolId::Term(3)],
                    vec![SymbolId::Term(4), SymbolId::NTerm(2)],
                ],
                reducer: Box::new(reducer.clone()),
            },
            Rule {
                words: vec![vec![SymbolId::Term(3)], vec![SymbolId::Term(5)]],
                reducer: Box::new(reducer.clone()),
            },
        ];
        assert_eq!(Ok(removed), remove_common(rules));
    }

    #[test]
    fn test_all_firsts() {
        let reducer = |_: &mut Vec<ReduceSymbol<usize, usize>>| {};
        let removed = vec![
            Rule {
                words: vec![
                    vec![SymbolId::NTerm(1), SymbolId::NTerm(2)],
                    vec![SymbolId::NTerm(3), SymbolId::NTerm(1)],
                ],
                reducer: Box::new(reducer.clone()),
            },
            Rule {
                words: vec![vec![SymbolId::Term(0)], vec![]],
                reducer: Box::new(reducer.clone()),
            },
            Rule {
                words: vec![vec![SymbolId::Term(1)], vec![]],
                reducer: Box::new(reducer.clone()),
            },
            Rule {
                words: vec![vec![SymbolId::Term(2)]],
                reducer: Box::new(reducer.clone()),
            },
        ];
        let expected = vec![
            vec![
                vec![
                    SymbolSet {
                        set: set! {0, 1},
                        has_eps: true,
                    },
                    SymbolSet {
                        set: set! {1},
                        has_eps: true,
                    },
                    SymbolSet {
                        set: set! {},
                        has_eps: true,
                    },
                ],
                vec![
                    SymbolSet {
                        set: set! {2},
                        has_eps: false,
                    },
                    SymbolSet {
                        set: set! {0},
                        has_eps: true,
                    },
                    SymbolSet {
                        set: set! {},
                        has_eps: true,
                    },
                ],
            ],
            vec![
                vec![
                    SymbolSet {
                        set: set! {0},
                        has_eps: false,
                    },
                    SymbolSet {
                        set: set! {},
                        has_eps: true,
                    },
                ],
                vec![SymbolSet {
                    set: set! {},
                    has_eps: true,
                }],
            ],
            vec![
                vec![
                    SymbolSet {
                        set: set! {1},
                        has_eps: false,
                    },
                    SymbolSet {
                        set: set! {},
                        has_eps: true,
                    },
                ],
                vec![SymbolSet {
                    set: set! {},
                    has_eps: true,
                }],
            ],
            vec![vec![
                SymbolSet {
                    set: set! {2},
                    has_eps: false,
                },
                SymbolSet {
                    set: set! {},
                    has_eps: true,
                },
            ]],
        ];
        assert_eq!(all_firsts(&removed), expected);
    }

    #[test]
    fn test_first() {
        let reducer = |_: &mut Vec<ReduceSymbol<usize, usize>>| {};
        let rules = map! {
            NTerm::Expr.id()=>
                Rule::<usize, usize> {
                    words: vec![
                        vec![NTerm::Term.id()],
                        vec![NTerm::Term.id(), Term::Add.id(), NTerm::Expr.id()],
                    ],
                    reducer: Box::new(reducer.clone()),
                },
            NTerm::Term.id()=>
                Rule {
                    words: vec![
                        vec![NTerm::Factor.id()],
                        vec![NTerm::Factor.id(), Term::Mul.id(), NTerm::Term.id()],
                    ],
                    reducer: Box::new(reducer.clone()),
                },
            NTerm::Factor.id()=>
                Rule {
                    words: vec![
                        vec![Term::Num(0).id()],
                        vec![Term::LP.id(), NTerm::Expr.id(), Term::RP.id()],
                    ],
                    reducer: Box::new(reducer.clone()),
                }
        };
        let removed = remove_common(rules).unwrap();
        let firsts = firsts(removed);
        let expected = vec![
            vec![SymbolSet {
                has_eps: false,
                set: set! {Term::Num(0).cardinal(), Term::LP.cardinal()},
            }],
            vec![SymbolSet {
                has_eps: false,
                set: set! {Term::Num(0).cardinal(), Term::LP.cardinal()},
            }],
            vec![
                SymbolSet {
                    has_eps: false,
                    set: set! {Term::Num(0).cardinal()},
                },
                SymbolSet {
                    has_eps: false,
                    set: set! {Term::LP.cardinal()},
                },
            ],
            vec![
                SymbolSet {
                    has_eps: true,
                    set: HashSet::new(),
                },
                SymbolSet {
                    has_eps: false,
                    set: set! {Term::Add.cardinal()},
                },
            ],
            vec![
                SymbolSet {
                    has_eps: true,
                    set: HashSet::new(),
                },
                SymbolSet {
                    has_eps: false,
                    set: set! {Term::Mul.cardinal()},
                },
            ],
        ];
        assert_eq!(firsts, expected);
    }

    #[test]
    fn test_follows() {
        let reducer = |_: &mut Vec<ReduceSymbol<usize, usize>>| {};
        let rules = map! {
            NTerm::Expr.id()=>
                Rule::<usize, usize> {
                    words: vec![
                        vec![NTerm::Term.id()],
                        vec![NTerm::Term.id(), Term::Add.id(), NTerm::Expr.id()],
                    ],
                    reducer: Box::new(reducer.clone()),
                },
            NTerm::Term.id()=>
                Rule {
                    words: vec![
                        vec![NTerm::Factor.id()],
                        vec![NTerm::Factor.id(), Term::Mul.id(), NTerm::Term.id()],
                    ],
                    reducer: Box::new(reducer.clone()),
                },
            NTerm::Factor.id()=>
                Rule {
                    words: vec![
                        vec![Term::Num(0).id()],
                        vec![Term::LP.id(), NTerm::Expr.id(), Term::RP.id()],
                    ],
                    reducer: Box::new(reducer.clone()),
                }
        };
        let removed = remove_common(rules).unwrap();
        let follows = follows(removed);
        let expected = vec![
            set! {Term::RP.cardinal()},
            set! {Term::Add.cardinal(), Term::RP.cardinal()},
            set! {Term::Add.cardinal(), Term::Mul.cardinal(), Term::RP.cardinal()},
            set! {Term::RP.cardinal()},
            set! {Term::RP.cardinal(), Term::Add.cardinal()},
        ];
        assert_eq!(follows, expected);
    }
}

#[derive(Clone, PartialEq, Debug)]
struct SymbolSet {
    set: HashSet<usize>,
    has_eps: bool,
}
impl Default for SymbolSet {
    fn default() -> Self {
        Self {
            set: HashSet::new(),
            has_eps: false,
        }
    }
}

type AllFirstSet = Vec<Vec<Vec<SymbolSet>>>;
type FirstSet = Vec<Vec<SymbolSet>>;
type FollowSet = Vec<HashSet<usize>>;

fn all_firsts<T, Ast>(rules: &IRules<T, Ast>) -> AllFirstSet {
    let mut fiw = Vec::new();
    let mut fia = Vec::new();
    // initialize
    for rule in rules {
        fia.push(SymbolSet::default());
        let mut set_for_words = Vec::new();
        for word in &rule.words {
            let mut set_for_slice = Vec::new();
            for _ in 0..word.len() {
                set_for_slice.push(SymbolSet::default());
            }
            set_for_slice.push(SymbolSet {
                has_eps: true,
                set: HashSet::new(),
            });
            set_for_words.push(set_for_slice);
        }
        fiw.push(set_for_words);
    }
    // calcurate set
    let mut changed = true;
    while changed {
        changed = false;
        let fia_old = fia.clone();
        let fiw_old = fiw.clone();
        for (a_idx, rule) in rules.iter().enumerate() {
            for (w_idx, words) in rule.words.iter().enumerate() {
                let mut has_eps = true;
                let mut fiw_acc = HashSet::new();
                for (s_idx, w) in words.iter().enumerate().rev() {
                    let fiw_i = &mut fiw[a_idx][w_idx][s_idx];
                    if let SymbolId::NTerm(id) = w {
                        if fia[*id].has_eps {
                            fiw_i.set = fiw_i
                                .set
                                .union(&fiw_acc)
                                .cloned()
                                .collect::<HashSet<usize>>();
                        } else {
                            has_eps = false;
                        }
                        fiw_i.set = fiw_i
                            .set
                            .union(&fia[*id].set)
                            .cloned()
                            .collect::<HashSet<usize>>();
                        fiw_acc = fiw_i.set.clone();
                    } else if let SymbolId::Term(id) = w {
                        fiw_acc = HashSet::new();
                        fiw_acc.insert(*id);
                        fiw_i.set = fiw_acc.clone();
                        has_eps = false;
                    }
                    fiw_i.has_eps = has_eps;
                }
                if let Some(ss) = &fiw[a_idx][w_idx].get(0) {
                    fia[a_idx].has_eps |= ss.has_eps;
                    fia[a_idx].set = fia[a_idx]
                        .set
                        .union(&ss.set)
                        .cloned()
                        .collect::<HashSet<usize>>();
                } else {
                    fia[a_idx].has_eps = true;
                }
            }
        }
        // ここの実装をサボっています
        changed |= fia_old != fia;
        changed |= fiw_old != fiw;
    }
    fiw
}

fn firsts<T, Ast>(rules: IRules<T, Ast>) -> FirstSet {
    all_firsts(&rules)
        .iter()
        .map(|per_rule| {
            per_rule
                .iter()
                .map(|per_word| per_word[0].clone())
                .collect::<Vec<SymbolSet>>()
        })
        .collect::<Vec<Vec<SymbolSet>>>()
}

/// (id of non-terminal symbol, id of rule in the non-terminal symbol) -> id of terminal symbol
fn follows<T, Ast>(rules: IRules<T, Ast>) -> FollowSet {
    let mut fo = FollowSet::new();
    let mut changed = true;
    fo.resize_with(rules.len(), || Default::default());
    let firsts = all_firsts(&rules);
    while changed {
        let fo_old = fo.clone();
        for (a_idx, rule) in rules.iter().enumerate() {
            for (w_idx, word) in rule.words.iter().enumerate() {
                for (s_idx, id) in word.iter().enumerate() {
                    if let SymbolId::NTerm(id) = id {
                        fo[*id] = fo[*id]
                            .union(&firsts[a_idx][w_idx][s_idx + 1].set)
                            .cloned()
                            .collect::<HashSet<usize>>();
                        if firsts[a_idx][w_idx][s_idx + 1].has_eps {
                            fo[*id] = fo[*id]
                                .union(&fo[a_idx])
                                .cloned()
                                .collect::<HashSet<usize>>();
                        }
                    }
                }
            }
        }
        changed = fo_old != fo;
    }
    fo
}

pub fn ll1<T: Terminal, NT: NonTerminal, Ast, F>(
    top: NT,
    eof: T,
    rules: Rule<T, Ast>,
    input: Vec<T>,
) -> Result<Ast, Error>
where
    F: FnMut(&mut Vec<ReduceSymbol<T, Ast>>),
{
    Err(Error::SyntaxError)
}
