#!/usr/bin/env bash

$STRIPE_DATA_HOME/bin/stripe-data "$@" \
  --model-file model.zip \
  --result-file result.csv \
  --growth-condition count-and-amount-2-.7-.7-10 \
  --feature count \
  --feature amount \
  --feature amount-factor-0.1 \
  --feature amount-factor-0.2 \
  --feature amount-factor-0.3 \
  --feature amount-factor-0.4 \
  --feature amount-factor-0.5 \
  --feature amount-factor-0.6 \
  --feature amount-factor-0.7 \
  --feature amount-factor-0.8 \
  --feature amount-factor-0.9 \
  --feature amount-factor-1.0 \
  --feature amount-factor-2.0 \
  --feature amount-factor-3.0 \
  --feature amount-factor-4.0 \
  --feature amount-factor-5.0 \
  --feature amount-factor-6.0 \
  --feature amount-factor-7.0 \
  --feature amount-factor-8.0 \
  --feature amount-factor-9.0 \
  --feature amount-factor-10.0 \
  --feature count-factor-0.1 \
  --feature count-factor-0.2 \
  --feature count-factor-0.3 \
  --feature count-factor-0.4 \
  --feature count-factor-0.5 \
  --feature count-factor-0.6 \
  --feature count-factor-0.7 \
  --feature count-factor-0.8 \
  --feature count-factor-0.9 \
  --feature count-factor-1.0 \
  --feature count-factor-2.0 \
  --feature count-factor-3.0 \
  --feature count-factor-4.0 \
  --feature count-factor-5.0 \
  --feature count-factor-6.0 \
  --feature count-factor-7.0 \
  --feature count-factor-8.0 \
  --feature count-factor-9.0 \
  --feature count-factor-10.0 \
