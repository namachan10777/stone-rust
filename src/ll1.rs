use std::collections::HashMap;
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
    RuleMustBeNonTerminal,
}

pub enum ReduceSymbol<T, Ast> {
    Term(T),
    Ast(Ast),
}

pub type Words = Vec<Vec<SymbolId>>;
pub struct Rule<T, Ast> {
    words: Words,
    reducer: Rc<Box<dyn FnOnce(&mut Vec<ReduceSymbol<T, Ast>>)>>,
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
fn remove_common_impl<T, Ast>(rules: &mut IRules<T, Ast>, target_idx: usize) {
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
                        reducer: Rc::new(Box::new(|_: &mut Vec<ReduceSymbol<T, Ast>>| {})),
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
                reducer: rules[target_idx].reducer.clone(),
            });
            remove_common_impl(rules, rules.len() - 1);
        }
        common_removed_rule.push(replaced);
    } else {
        common_removed_rule.push(rules[target_idx].words[begin].clone());
    }
    rules[target_idx] = Rule {
        words: common_removed_rule,
        reducer: rules[target_idx].reducer.clone(),
    };
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
        let mut rules = HashMap::new();
        let reducer = |_: &mut Vec<ReduceSymbol<usize, usize>>| {};
        rules.insert(
            NTerm::Expr.id(),
            Rule::<usize, usize> {
                words: vec![
                    vec![NTerm::Term.id()],
                    vec![NTerm::Term.id(), Term::Add.id(), NTerm::Expr.id()],
                ],
                reducer: Rc::new(Box::new(reducer.clone())),
            },
        );
        rules.insert(
            NTerm::Term.id(),
            Rule {
                words: vec![
                    vec![NTerm::Factor.id()],
                    vec![NTerm::Factor.id(), Term::Mul.id(), NTerm::Term.id()],
                ],
                reducer: Rc::new(Box::new(reducer.clone())),
            },
        );
        rules.insert(
            NTerm::Factor.id(),
            Rule {
                words: vec![
                    vec![Term::Num(0).id()],
                    vec![Term::LP.id(), NTerm::Expr.id(), Term::RP.id()],
                ],
                reducer: Rc::new(Box::new(reducer.clone())),
            },
        );
        let removed = vec![
            (Rule {
                words: vec![vec![NTerm::Term.id(), SymbolId::NTerm(3)]],
                reducer: Rc::new(Box::new(reducer.clone())),
            }),
            (Rule {
                words: vec![vec![NTerm::Factor.id(), SymbolId::NTerm(4)]],
                reducer: Rc::new(Box::new(reducer.clone())),
            }),
            (Rule {
                words: vec![
                    vec![Term::Num(0).id()],
                    vec![Term::LP.id(), NTerm::Expr.id(), Term::RP.id()],
                ],
                reducer: Rc::new(Box::new(reducer.clone())),
            }),
            (Rule {
                words: vec![vec![], vec![Term::Add.id(), NTerm::Expr.id()]],
                reducer: Rc::new(Box::new(reducer.clone())),
            }),
            (Rule {
                words: vec![vec![], vec![Term::Mul.id(), NTerm::Term.id()]],
                reducer: Rc::new(Box::new(reducer.clone())),
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
                reducer: Rc::new(Box::new(reducer.clone())),
            },
        );
        let removed = vec![
            Rule {
                words: vec![
                    vec![SymbolId::Term(0), SymbolId::Term(2), SymbolId::Term(3)],
                    vec![SymbolId::Term(1), SymbolId::NTerm(1)],
                ],
                reducer: Rc::new(Box::new(reducer.clone())),
            },
            Rule {
                words: vec![
                    vec![SymbolId::Term(2), SymbolId::Term(3)],
                    vec![SymbolId::Term(4), SymbolId::NTerm(2)],
                ],
                reducer: Rc::new(Box::new(reducer.clone())),
            },
            Rule {
                words: vec![vec![SymbolId::Term(3)], vec![SymbolId::Term(5)]],
                reducer: Rc::new(Box::new(reducer.clone())),
            },
        ];
        assert_eq!(Ok(removed), remove_common(rules));
    }
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
