# syntax=docker/dockerfile:1
FROM elixir:1.16-alpine AS builder

WORKDIR /app
RUN apk add --no-cache build-base git python3 py3-pip py3-numpy py3-scikit-learn

COPY mix.exs mix.lock ./
RUN mix local.hex --force && mix local.rebar --force
RUN MIX_ENV=prod mix deps.get --only prod
RUN MIX_ENV=prod mix deps.compile

COPY lib ./lib
COPY config ./config
COPY resources ./resources
COPY convert_data.py ./
COPY native ./native

RUN python3 convert_data.py resources/references.json.gz

RUN cd native/knn && make

RUN MIX_ENV=prod mix release --overwrite

FROM elixir:1.16-alpine AS runtime

WORKDIR /app

RUN apk add --no-cache libgcc

COPY resources/normalization.json resources/mcc_risk.json ./resources/
COPY --from=builder /app/resources/vectors_8d_sorted.bin /app/resources/labels_sorted.bin ./resources/
COPY --from=builder /app/resources/centroids.bin /app/resources/bucket_starts.bin /app/resources/svd_matrix.bin ./resources/
COPY --from=builder /app/resources/lda_w.bin /app/resources/lda_w0.bin ./resources/
COPY --from=builder /app/resources/fraud_centroid.bin /app/resources/legit_centroid.bin /app/resources/cov_inv.bin ./resources/
COPY --from=builder /app/native/knn/priv/knn.so ./priv/
COPY --from=builder /app/native/knn/priv/knn.so ./lib/rinha_fraud-0.1.0/priv/

COPY --from=builder /app/_build/prod/rel/rinha_fraud ./

ENV MIX_ENV=prod

CMD ["./bin/rinha_fraud", "start"]
