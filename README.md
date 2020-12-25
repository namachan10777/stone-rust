# stone-rust

[![GitHub Actions](https://github.com/namachan10777/stone-rust/workflows/Rust/badge.svg)](https://github.com/namachan10777/stone-rust/actions?query=workflow:Rust)

Compiler for stone language.

# 実装内容
課題3について実装した。依存ライブラリはコマンドライン引数のパーサのみである。
実行は以下のように行える。
今回は共通部分を自動で除去するLL(1)パーサジェネレータ及びそれを使った課題の実装のみまでしか出来なかったが、
時間がある時にVMでの実行とx86\_64向けバイナリの生成まで実装したい。
パーサジェネレータを作成する際に記号を還元する関数をどう持ち回るを自分で実装しながら考えられたので良い経験になった。
```sh
cargo run -- ./example/test2.stone
```
