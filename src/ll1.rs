use std::collections::{HashMap, HashSet};
use std::fmt;
use std::rc::Rc;

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
    MayBeLeftCyclic,
    ThisIsNotLL1,
    RuleMustBeNonTerminal,
}

pub enum ReduceSymbol<T, Ast> {
    Term(T),
    Ast(Ast),
}

pub type Words = Vec<SymbolId>;
type Reducer<T, Ast> = Rc<Box<dyn Fn(&mut Vec<ReduceSymbol<T, Ast>>)>>;
pub struct Rule<T, Ast> {
    words: Words,
    reducer: Reducer<T, Ast>,
}
impl<T, Ast> Clone for Rule<T, Ast> {
    fn clone(&self) -> Self {
        Self {
            words: self.words.clone(),
            reducer: self.reducer.clone(),
        }
    }
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

enum IReducer<T, Ast> {
    Tag(usize),
    Root(HashMap<Vec<usize>, Reducer<T, Ast>>),
}

impl<T, Ast> Clone for IReducer<T, Ast> {
    fn clone(&self) -> Self {
        match self {
            IReducer::Tag(id) => IReducer::Tag(*id),
            IReducer::Root(map) => IReducer::Root(map.clone()),
        }
    }
}

impl<T, Ast> fmt::Debug for IReducer<T, Ast> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            IReducer::Tag(id) => f.write_fmt(format_args!("Tag({})", *id)),
            IReducer::Root(map) => f.debug_list().entries(map.keys()).finish(),
        }
    }
}

impl<T, Ast> PartialEq for IReducer<T, Ast> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (IReducer::Tag(lid), IReducer::Tag(rid)) => lid == rid,
            (IReducer::Root(lmap), IReducer::Root(rmap)) => lmap.keys().eq(rmap.keys()),
            _ => false,
        }
    }
}

struct IRule<T, Ast> {
    words: Vec<SymbolId>,
    reducer: IReducer<T, Ast>,
}
impl<T,Ast> fmt::Debug for IRule<T,Ast> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        f.debug_struct("IRule").field("words", &self.words).field("reducer", &self.reducer).finish()
    }
}

impl<T,Ast> Clone for IRule<T,Ast> {
    fn clone(&self) -> Self {
        Self {
            words: self.words.clone(),
            reducer: self.reducer.clone(),
        }
    }
}

impl<T,Ast> PartialEq for IRule<T,Ast> {
    fn eq(&self, other: &Self) -> bool {
        self.words.eq(&other.words) && self.reducer.eq(&other.reducer)
    }
}

// key is a cardinal of non-terminal symbol
pub type Rules<T, Ast> = HashMap<usize, Vec<Rule<T, Ast>>>;
pub type IRules<T, Ast> = Vec<Vec<IRule<T, Ast>>>;

fn gen_fluxed_and_lasts_rule<T, Ast>(
    rules: &[Rule<T, Ast>],
) -> (IRule<T, Ast>, Option<Vec<IRule<T, Ast>>>) {
    // FIXME
    let closure = |_: &mut Vec<ReduceSymbol<T, Ast>>| {};
    let noop: Reducer<T, Ast> = Rc::new(Box::new(closure.clone()));
    let mut until_common_idx = 0;
    'linear_check: loop {
        let mut sample = rules[0].words.get(until_common_idx);
        for rule in rules {
            if rule.words.len() == until_common_idx || sample != rule.words.get(until_common_idx) {
                break 'linear_check;
            }
        }
        until_common_idx += 1;
        sample = rules[0].words.get(until_common_idx);
    }
    let common = Rule {
        reducer: rules[0].reducer.clone(),
        words: rules[0].words[0..until_common_idx].to_vec(),
    };
    let tails = rules.iter().map(|rule| Rule {
        reducer: noop.clone(),
        words: rule.words[until_common_idx..].to_owned(),
    });
    if tails.clone().all(|rule| rule.words.len() == 0) {
        (common, None)
    } else {
        (common, Some(tails.collect::<Vec<IRule<T, Ast>>>()))
    }
}

// internal impl
// 先頭が共通していれば共通部分単体のルールに書き換え、共通以後を新ルールにして追加、新ルールにも適用
fn remove_common_impl<'a, T, Ast>(rules: &mut IRules<T, Ast>, target_idx: usize) {
    rules[target_idx].sort_by(|rule1, rule2| {
        if rule1.words.is_empty() {
            Ordering::Less
        } else if rule2.words.is_empty() {
            Ordering::Greater
        } else {
            rule1.words[0].partial_cmp(&rule2.words[0]).unwrap()
        }
    });
    let mut begin = 0;
    let mut common_removed_rule = Vec::new();
    for i in 0..rules[target_idx].len() {
        if rules[target_idx][begin].words.get(0) != rules[target_idx][i].words.get(0) {
            // 先頭を同じくするコードが複数ある場合
            if i - begin > 1 {
                let (mut replaced, new_rule) =
                    gen_fluxed_and_lasts_rule(&rules[target_idx][begin..i]);
                if let Some(new_rule) = new_rule {
                    replaced.words.push(SymbolId::NTerm(rules.len()));
                    rules.push(new_rule);
                    remove_common_impl(rules, rules.len() - 1);
                }
                common_removed_rule.push(replaced);
            } else {
                common_removed_rule.push(rules[target_idx][begin].clone());
            }
            begin = i;
        }
    }
    if rules[target_idx].len() - begin > 1 {
        let (mut replaced, new_rule) =
            gen_fluxed_and_lasts_rule(&rules[target_idx][begin..rules[target_idx].len()]);
        if let Some(new_rule) = new_rule {
            replaced.words.push(SymbolId::NTerm(rules.len()));
            rules.push(new_rule);
            remove_common_impl(rules, rules.len() - 1);
        }
        common_removed_rule.push(replaced);
    } else {
        common_removed_rule.push(rules[target_idx][begin].clone());
    }
    rules[target_idx] = common_removed_rule;
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
        .collect::<Vec<(usize, Vec<Rule<T, Ast>>)>>();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    let mut tbl = pairs
        .into_iter()
        .map(|(_, rule)| rule)
        .collect::<Vec<Vec<Rule<T, Ast>>>>();
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
                Term::Add => 0,
                Term::Mul => 1,
                Term::Num(_) => 2,
                Term::LP => 3,
                Term::RP => 4,
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
        let dummy: Reducer<(), ()> = Rc::new(Box::new(|_: &mut Vec<ReduceSymbol<(), ()>>| {}));
        let (common, tails) = gen_fluxed_and_lasts_rule(&[
            Rule {
                words: vec![
                    SymbolId::NTerm(0),
                    SymbolId::NTerm(0),
                    SymbolId::NTerm(0),
                    SymbolId::NTerm(0),
                ],
                reducer: dummy.clone(),
            },
            Rule {
                words: vec![
                    SymbolId::NTerm(0),
                    SymbolId::NTerm(0),
                    SymbolId::NTerm(1),
                    SymbolId::NTerm(0),
                ],
                reducer: dummy.clone(),
            },
            Rule {
                words: vec![
                    SymbolId::NTerm(0),
                    SymbolId::NTerm(0),
                    SymbolId::NTerm(1),
                    SymbolId::NTerm(2),
                ],
                reducer: dummy.clone(),
            },
        ]);
        assert_eq!(
            common,
            Rule {
                words: vec![SymbolId::NTerm(0), SymbolId::NTerm(0)],
                reducer: dummy.clone()
            }
        );
        assert_eq!(
            tails,
            Some(vec![
                Rule {
                    words: vec![SymbolId::NTerm(0), SymbolId::NTerm(0)],
                    reducer: dummy.clone()
                },
                Rule {
                    words: vec![SymbolId::NTerm(1), SymbolId::NTerm(0)],
                    reducer: dummy.clone()
                },
                Rule {
                    words: vec![SymbolId::NTerm(1), SymbolId::NTerm(2)],
                    reducer: dummy.clone()
                },
            ])
        );
        let (common, tails) = gen_fluxed_and_lasts_rule(&[
            Rule {
                words: vec![
                    SymbolId::NTerm(0),
                    SymbolId::NTerm(0),
                    SymbolId::NTerm(0),
                    SymbolId::NTerm(0),
                ],
                reducer: dummy.clone(),
            },
            Rule {
                words: vec![
                    SymbolId::NTerm(0),
                    SymbolId::NTerm(0),
                    SymbolId::NTerm(0),
                    SymbolId::NTerm(0),
                ],
                reducer: dummy.clone(),
            },
            Rule {
                words: vec![
                    SymbolId::NTerm(0),
                    SymbolId::NTerm(0),
                    SymbolId::NTerm(0),
                    SymbolId::NTerm(0),
                ],
                reducer: dummy.clone(),
            },
        ]);
        assert_eq!(
            common,
            Rule {
                words: vec![
                    SymbolId::NTerm(0),
                    SymbolId::NTerm(0),
                    SymbolId::NTerm(0),
                    SymbolId::NTerm(0)
                ],
                reducer: dummy.clone()
            }
        );
        assert_eq!(tails, None);
    }

    #[test]
    fn test_remove_common() {
        let reducer: Reducer<usize, usize> =
            Rc::new(Box::new(|_: &mut Vec<ReduceSymbol<usize, usize>>| {}));
        let rules = map! {
            NTerm::Expr.cardinal()=> vec![
                Rule { words: vec![NTerm::Term.id()], reducer: reducer.clone()},
                Rule { words: vec![NTerm::Term.id(), Term::Add.id(), NTerm::Expr.id()], reducer: reducer.clone()}
            ],
            NTerm::Term.cardinal()=>vec![
                 Rule { words: vec![NTerm::Factor.id()], reducer: reducer.clone()},
                 Rule { words: vec![NTerm::Factor.id(), Term::Mul.id(), NTerm::Term.id()], reducer: reducer.clone()}
            ],
            NTerm::Factor.cardinal()=>vec![
                  Rule { words: vec![Term::Num(0).id()], reducer: reducer.clone()},
                  Rule { words: vec![Term::LP.id(), NTerm::Expr.id(), Term::RP.id()], reducer: reducer.clone()}
            ]
        };
        let removed = vec![
            vec![Rule {
                words: vec![NTerm::Term.id(), SymbolId::NTerm(3)],
                reducer: reducer.clone(),
            }],
            vec![Rule {
                words: vec![NTerm::Factor.id(), SymbolId::NTerm(4)],
                reducer: reducer.clone(),
            }],
            vec![
                Rule {
                    words: vec![Term::Num(0).id()],
                    reducer: reducer.clone(),
                },
                Rule {
                    words: vec![Term::LP.id(), NTerm::Expr.id(), Term::RP.id()],
                    reducer: reducer.clone(),
                },
            ],
            vec![
                Rule {
                    words: vec![],
                    reducer: reducer.clone(),
                },
                Rule {
                    words: vec![Term::Add.id(), NTerm::Expr.id()],
                    reducer: reducer.clone(),
                },
            ],
            vec![
                Rule {
                    words: vec![],
                    reducer: reducer.clone(),
                },
                Rule {
                    words: vec![Term::Mul.id(), NTerm::Term.id()],
                    reducer: reducer.clone(),
                },
            ],
        ];
        assert_eq!(Ok(removed), remove_common(rules));
        let mut rules = HashMap::new();
        rules.insert(
            0,
            vec![
                Rule {
                    words: vec![SymbolId::Term(1), SymbolId::Term(2), SymbolId::Term(3)],
                    reducer: reducer.clone(),
                },
                Rule {
                    words: vec![SymbolId::Term(1), SymbolId::Term(4), SymbolId::Term(3)],
                    reducer: reducer.clone(),
                },
                Rule {
                    words: vec![SymbolId::Term(1), SymbolId::Term(4), SymbolId::Term(5)],
                    reducer: reducer.clone(),
                },
                Rule {
                    words: vec![SymbolId::Term(1), SymbolId::Term(4), SymbolId::Term(5)],
                    reducer: reducer.clone(),
                },
                Rule {
                    words: vec![SymbolId::Term(0), SymbolId::Term(2), SymbolId::Term(3)],
                    reducer: reducer.clone(),
                },
            ],
        );
        let removed = vec![
            vec![
                Rule {
                    words: vec![SymbolId::Term(0), SymbolId::Term(2), SymbolId::Term(3)],
                    reducer: reducer.clone(),
                },
                Rule {
                    words: vec![SymbolId::Term(1), SymbolId::NTerm(1)],
                    reducer: reducer.clone(),
                },
            ],
            vec![
                Rule {
                    words: vec![SymbolId::Term(2), SymbolId::Term(3)],
                    reducer: reducer.clone(),
                },
                Rule {
                    words: vec![SymbolId::Term(4), SymbolId::NTerm(2)],
                    reducer: reducer.clone(),
                },
            ],
            vec![
                Rule {
                    words: vec![SymbolId::Term(3)],
                    reducer: reducer.clone(),
                },
                Rule {
                    words: vec![SymbolId::Term(5)],
                    reducer: reducer.clone(),
                },
            ],
        ];
        assert_eq!(Ok(removed), remove_common(rules));
    }

    #[test]
    fn test_all_firsts() {
        let reducer: Reducer<(), ()> = Rc::new(Box::new(|_: &mut Vec<ReduceSymbol<(), ()>>| {}));
        let removed = vec![
            vec![
                Rule {
                    words: vec![SymbolId::NTerm(1), SymbolId::NTerm(2)],
                    reducer: reducer.clone(),
                },
                Rule {
                    words: vec![SymbolId::NTerm(3), SymbolId::NTerm(1)],
                    reducer: reducer.clone(),
                },
            ],
            vec![
                Rule {
                    words: vec![SymbolId::Term(0)],
                    reducer: reducer.clone(),
                },
                Rule {
                    words: vec![],
                    reducer: reducer.clone(),
                },
            ],
            vec![
                Rule {
                    words: vec![SymbolId::Term(1)],
                    reducer: reducer.clone(),
                },
                Rule {
                    words: vec![],
                    reducer: reducer.clone(),
                },
            ],
            vec![Rule {
                words: vec![SymbolId::Term(2)],
                reducer: reducer.clone(),
            }],
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
        assert_eq!(all_firsts(&removed).unwrap(), expected);
    }

    #[test]
    fn test_first() {
        let reducer: Reducer<(), ()> = Rc::new(Box::new(|_: &mut Vec<ReduceSymbol<(), ()>>| {}));
        let rules = map! {
            NTerm::Expr.cardinal()=> vec![
                Rule { words: vec![NTerm::Term.id()], reducer: reducer.clone()},
                Rule { words: vec![NTerm::Term.id(), Term::Add.id(), NTerm::Expr.id()], reducer: reducer.clone()}
            ],
            NTerm::Term.cardinal()=>vec![
                 Rule { words: vec![NTerm::Factor.id()], reducer: reducer.clone()},
                 Rule { words: vec![NTerm::Factor.id(), Term::Mul.id(), NTerm::Term.id()], reducer: reducer.clone()}
            ],
            NTerm::Factor.cardinal()=>vec![
                  Rule { words: vec![Term::Num(0).id()], reducer: reducer.clone()},
                  Rule { words: vec![Term::LP.id(), NTerm::Expr.id(), Term::RP.id()], reducer: reducer.clone()}
            ]
        };
        let removed = remove_common(rules).unwrap();
        let firsts = firsts(&removed).unwrap();
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
        let reducer: Reducer<(), ()> = Rc::new(Box::new(|_: &mut Vec<ReduceSymbol<(), ()>>| {}));
        let rules = map! {
            NTerm::Expr.cardinal()=> vec![
                Rule { words: vec![NTerm::Term.id()], reducer: reducer.clone()},
                Rule { words: vec![NTerm::Term.id(), Term::Add.id(), NTerm::Expr.id()], reducer: reducer.clone()}
            ],
            NTerm::Term.cardinal()=>vec![
                 Rule { words: vec![NTerm::Factor.id()], reducer: reducer.clone()},
                 Rule { words: vec![NTerm::Factor.id(), Term::Mul.id(), NTerm::Term.id()], reducer: reducer.clone()}
            ],
            NTerm::Factor.cardinal()=>vec![
                  Rule { words: vec![Term::Num(0).id()], reducer: reducer.clone()},
                  Rule { words: vec![Term::LP.id(), NTerm::Expr.id(), Term::RP.id()], reducer: reducer.clone()}
            ]
        };
        let removed = remove_common(rules).unwrap();
        let follows = follows(&removed).unwrap();
        let expected = vec![
            set! {Term::RP.cardinal()},
            set! {Term::Add.cardinal(), Term::RP.cardinal()},
            set! {Term::Add.cardinal(), Term::Mul.cardinal(), Term::RP.cardinal()},
            set! {Term::RP.cardinal()},
            set! {Term::RP.cardinal(), Term::Add.cardinal()},
        ];
        assert_eq!(follows, expected);
    }

    #[test]
    fn test_gen_table() {
        let reducer: Reducer<Term, ()> =
            Rc::new(Box::new(|_: &mut Vec<ReduceSymbol<Term, ()>>| {}));
        let rules = map! {
            NTerm::Expr.cardinal()=> vec![
                Rule { words: vec![NTerm::Term.id()], reducer: reducer.clone()},
                Rule { words: vec![NTerm::Term.id(), Term::Add.id(), NTerm::Expr.id()], reducer: reducer.clone()}
            ],
            NTerm::Term.cardinal()=>vec![
                 Rule { words: vec![NTerm::Factor.id()], reducer: reducer.clone()},
                 Rule { words: vec![NTerm::Factor.id(), Term::Mul.id(), NTerm::Term.id()], reducer: reducer.clone()}
            ],
            NTerm::Factor.cardinal()=>vec![
                  Rule { words: vec![Term::Num(0).id()], reducer: reducer.clone()},
                  Rule { words: vec![Term::LP.id(), NTerm::Expr.id(), Term::RP.id()], reducer: reducer.clone()}
            ]
        };
        let removed = remove_common(rules).unwrap();
        let tbl = gen_table(removed);
        let r_expr = Rule {
            words: vec![NTerm::Term.id(), SymbolId::NTerm(3)],
            reducer: reducer.clone(),
        };
        let r_term = Rule {
            words: vec![NTerm::Factor.id(), SymbolId::NTerm(4)],
            reducer: reducer.clone(),
        };
        let r_factor1 = Rule {
            words: vec![Term::Num(0).id()],
            reducer: reducer.clone(),
        };
        let r_factor2 = Rule {
            words: vec![Term::LP.id(), NTerm::Expr.id(), Term::RP.id()],
            reducer: reducer.clone(),
        };
        let r_emp = Rule {
            words: vec![],
            reducer: reducer.clone(),
        };
        let r_expr_2 = Rule {
            words: vec![Term::Add.id(), NTerm::Expr.id()],
            reducer: reducer.clone(),
        };
        let r_term_2 = Rule {
            words: vec![Term::Mul.id(), NTerm::Term.id()],
            reducer: reducer.clone(),
        };
        let expected = vec![
            vec![None, None, Some(r_expr.clone()), Some(r_expr.clone()), None],
            vec![None, None, Some(r_term.clone()), Some(r_term.clone()), None],
            vec![
                None,
                None,
                Some(r_factor1.clone()),
                Some(r_factor2.clone()),
                None,
            ],
            vec![
                Some(r_expr_2.clone()),
                None,
                None,
                None,
                Some(r_emp.clone()),
            ],
            vec![
                Some(r_emp.clone()),
                Some(r_term_2.clone()),
                None,
                None,
                Some(r_emp.clone()),
            ],
        ];
        assert_eq!(Ok(expected), tbl);
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

fn all_firsts<T, Ast>(rules: &IRules<T, Ast>) -> Result<AllFirstSet, Error> {
    let mut fiw = Vec::new();
    let mut fia = Vec::new();
    // initialize
    for nterm in rules {
        fia.push(SymbolSet::default());
        let mut set_for_words = Vec::new();
        for rule in nterm {
            let mut set_for_slice = Vec::new();
            for _ in 0..rule.words.len() {
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
    let mut count = 0;
    while changed {
        count += 1;
        if count > 10000 {
            return Err(Error::MayBeLeftCyclic);
        }
        changed = false;
        let fia_old = fia.clone();
        let fiw_old = fiw.clone();
        for (a_idx, nterm) in rules.iter().enumerate() {
            for (w_idx, rule) in nterm.iter().enumerate() {
                let mut has_eps = true;
                let mut fiw_acc = HashSet::new();
                for (s_idx, w) in rule.words.iter().enumerate().rev() {
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
    Ok(fiw)
}

fn firsts<T, Ast>(rules: &IRules<T, Ast>) -> Result<FirstSet, Error> {
    all_firsts(&rules).map(|firsts| {
        firsts
            .iter()
            .map(|per_rule| {
                per_rule
                    .iter()
                    .map(|per_word| per_word[0].clone())
                    .collect::<Vec<SymbolSet>>()
            })
            .collect::<Vec<Vec<SymbolSet>>>()
    })
}

/// (id of non-terminal symbol, id of rule in the non-terminal symbol) -> id of terminal symbol
fn follows<T, Ast>(rules: &IRules<T, Ast>) -> Result<FollowSet, Error> {
    let mut fo = FollowSet::new();
    let mut changed = true;
    fo.resize_with(rules.len(), || Default::default());
    let firsts = all_firsts(&rules)?;
    let mut count = 0;
    while changed {
        count += 1;
        if count > 10000 {
            return Err(Error::MayBeLeftCyclic);
        }
        let fo_old = fo.clone();
        for (a_idx, nterm) in rules.iter().enumerate() {
            for (w_idx, rule) in nterm.iter().enumerate() {
                for (s_idx, id) in rule.words.iter().enumerate() {
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
    Ok(fo)
}

type Table<T, Ast> = Vec<Vec<Option<Rule<T, Ast>>>>;

fn gen_table<T: Terminal, Ast>(rules: IRules<T, Ast>) -> Result<Table<T, Ast>, Error> {
    let mut tbl = Vec::new();
    tbl.resize_with(rules.len(), || Vec::new());
    tbl.iter_mut().for_each(|v| v.resize_with(T::N, || None));
    let firsts = firsts(&rules)?;
    let follows = follows(&rules)?;
    for nt_idx in 0..rules.len() {
        for t_idx in 0..T::N {
            for (w_idx, first_of_word) in firsts[nt_idx].iter().enumerate() {
                if first_of_word.set.contains(&t_idx)
                    || (first_of_word.has_eps && follows[nt_idx].contains(&t_idx))
                {
                    if tbl[nt_idx][t_idx].is_some() {
                        return Err(Error::ThisIsNotLL1);
                    } else {
                        tbl[nt_idx][t_idx] = Some(rules[nt_idx][w_idx].clone());
                    }
                }
            }
        }
    }
    Ok(tbl)
}

pub fn ll1<T: Terminal, NT: NonTerminal, Ast: Clone, F>(
    top: NT,
    eof: T,
    rules: Rules<T, Ast>,
    input: Vec<T>,
) -> Result<Ast, Error>
where
    F: FnMut(&mut Vec<ReduceSymbol<T, Ast>>),
{
    let removed = remove_common(rules)?;
    let tbl = gen_table(removed)?;
    let mut input_id = input
        .iter()
        .map(|tok| tok.cardinal())
        .collect::<Vec<usize>>();
    let mut ast_stack = input
        .into_iter()
        .filter_map(|tok| {
            if tok.accept() {
                Some(ReduceSymbol::Term(tok))
            } else {
                None
            }
        })
        .collect::<Vec<ReduceSymbol<T, Ast>>>();
    let mut reducer_stack: Vec<Reducer<T, Ast>> = Vec::new();
    let mut child_count: Vec<usize> = Vec::new();
    let mut state = Vec::new();
    state.push(eof.id());
    while !state.is_empty() {
        let top_state = state.pop().unwrap();
        if let SymbolId::Term(id) = top_state {
            if input_id.pop() != Some(id) {
                return Err(Error::SyntaxError);
            }
            while child_count.last() == Some(&1) {
                let reducer = reducer_stack.pop().unwrap();
                reducer(&mut ast_stack);
                child_count.pop();
            }
            if !child_count.is_empty() {
                let last_idx = child_count.len() - 1;
                child_count[last_idx] -= 1;
            }
        } else if let SymbolId::NTerm(id) = top_state {
            if let Some(rewrite) = &tbl[id][*input_id.last().ok_or(Error::SyntaxError)?] {
                child_count.push(rewrite.words.len());
                reducer_stack.push(rewrite.reducer.clone());
                for id in rewrite.words.iter().rev() {
                    state.push(id.clone());
                }
            } else {
                return Err(Error::SyntaxError);
            }
        }
    }
    if child_count.is_empty() && input_id.is_empty() {
        if ast_stack.len() == 1 {
            if let ReduceSymbol::Ast(ast) = &ast_stack[0] {
                return Ok(ast.clone());
            }
        }
    }
    return Err(Error::SyntaxError);
}
