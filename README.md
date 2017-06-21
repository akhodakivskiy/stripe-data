# Predicting high growth users for Stripe

This project allows users to train and evaluate models to predict high growth user. The entry point is the `StripeGrowthApp`. The user can execute the main program using the `stripe-growth-app` script (generaged by `sbt-native-assembly` plugin)

There is also sample execution script in `run_model.sh` which expects `STRIPE_DATA_HOME` enviroment variable to be set to the path of the project build created by `sbt stage` or `sbt universal:packageBin`:

```bash
$ sbt stage
...
$ STRIPE_DATA_HOME=./target/universal/stage ./run_model.sh --data-file <path to data file>
...
```

```
Usage: neural [options]

  --data-file <value>      path to the data file
  --growth-condition <value>
                           condition that high growth users must meet
  --feature <value>        transaction history feature
  --train-set-share <value>
                           training set share
  --test-set-share <value>
                           test set share
  --validate-set-share <value>
                           validate set share
  --min-transactions <value>
                           minimum number of transactions per user
  --random-seed <value>    random number generator seed
  --model-file <value>     path to model file
  --result-file <value>    path to to store evaluation results
  --eval-only              do not train the model, only evaluate using pre-trained model
```

The application works in two modes: training and evaluation. The latter is enabled by passing the `--eval-only` flag. The user can specify some constants as to how to split the data into training, testing, and validation sets. As well as specify the minimum number of transactions per user (other users will be discarded). The models can be saved to and restored from files using the `--model-file` option.

## The High Growth User

High growth user is one that's using the platform more as the time passes. This project allows to evaluate different high conditions. This is done by implementing the `GrowthCondition` trait.

```scala
trait GrowthCondition {
  def isHyperGrowth(txs: Seq[Transaction]): Boolean
}
```

Currently there are 3 implementations focusing on the (1) amount transacted, the (2) number of transactions and (3) both

- `amount-<period factor>-<growth factor>-<min transactions>`
- `count-<period factor>-<growth factor>-<min transactions>`
- `count-and-amount-<period factor>-<count growth factor>-<amount growth factor>-<min transactions>`

Here the transactions for a specific user are split into two groups based on the transaction time. If the first transation time is `TS`, the last transation time is `TE` then we are going to choose threshold time `TT` such that `(TE - TT) / (TT - TS)` equals to the `<period factor>`. In the example below `<period factor>` is roughly 2.

```
       period1      period2
-----TS-------TT--------------TE------> time
```

If we count the numbers of transactions and their total amounts in both periods we can now argue whether the user are high growth or no. Amount growth factor is the threshold for the ratio fo the total amount in the second period to the total amount in the first period. Count growth factor is defined similarly.

## Features of Users

We describe users using various features which serve as inputs to the model. This is done by implementing the `Feature` trait.

```scala
trait Feature {
  def value(txs: Seq[Transaction]): Double
}
```

Currently there are 4 different features implemented.

- `amount`
- `count`
- `amount-factor-<period factor>`
- `count-factor-<period factor>`

The first two just count the total amount and number of transactions for each user. The latter two calculate the ratio of the amounts or the counts in two periods defined by the `<period factor>` (see the explanation ablove)

These features describe users on multiple dimensions that hopefully correlate with our definition of the High Growth User.
