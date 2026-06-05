defmodule RinhaFraud.RandomForestTest do
  use ExUnit.Case, async: true

  test "loads sklearn-style binary forest and predicts leaf probability" do
    forest =
      <<
        1::little-signed-32,
        2::little-signed-32,
        3::little-signed-32,
        0::little-signed-32,
        0.5::float-little-32,
        1::little-signed-32,
        2::little-signed-32,
        0.0::float-little-32,
        -1::little-signed-32,
        0.0::float-little-32,
        -1::little-signed-32,
        -1::little-signed-32,
        0.1::float-little-32,
        -1::little-signed-32,
        0.0::float-little-32,
        -1::little-signed-32,
        -1::little-signed-32,
        0.9::float-little-32,
        0.5::float-little-32
      >>
      |> RinhaFraud.RandomForest.load()

    assert_in_delta RinhaFraud.RandomForest.predict_proba(forest, [0.4, 1.0]), 0.1, 0.00001
    assert_in_delta RinhaFraud.RandomForest.predict_proba(forest, [0.6, 1.0]), 0.9, 0.00001
    assert RinhaFraud.RandomForest.classify(forest, [0.4, 1.0]) == 0.0
    assert RinhaFraud.RandomForest.classify(forest, [0.6, 1.0]) == 1.0
  end
end
