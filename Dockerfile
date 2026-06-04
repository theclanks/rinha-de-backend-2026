# syntax=docker/dockerfile:1
FROM elixir:1.16-alpine AS builder

WORKDIR /app
RUN apk add --no-cache build-base git

COPY mix.exs mix.lock ./
RUN mix local.hex --force && mix local.rebar --force
RUN MIX_ENV=prod mix deps.get --only prod
RUN MIX_ENV=prod mix deps.compile

COPY lib ./lib
COPY config ./config
COPY native ./native

RUN cd native/knn && make clean && make ERTS_INCLUDE_DIR=/usr/local/lib/erlang/erts-14.2.5.15/include

RUN MIX_ENV=prod mix release --overwrite

FROM elixir:1.16-alpine AS runtime

WORKDIR /app

RUN apk add --no-cache libgcc

# Copy pre-processed binary files directly (no Python build step needed)
COPY resources/normalization.json resources/mcc_risk.json ./resources/
COPY resources/vectors_8d_sorted.bin resources/labels_sorted.bin ./resources/
COPY resources/centroids.bin resources/bucket_starts.bin resources/svd_matrix.bin ./resources/
COPY resources/lda_w.bin resources/lda_w0.bin ./resources/
COPY resources/fraud_centroid.bin resources/legit_centroid.bin resources/cov_inv.bin ./resources/
COPY resources/cart_tree.bin ./resources/

COPY --from=builder /app/_build/prod/rel/rinha_fraud ./

# Copy NIF to the correct priv directory inside the release
COPY --from=builder /app/native/knn/priv/knn.so /tmp/knn.so
RUN mkdir -p ./lib/rinha_fraud-0.1.0/priv && cp /tmp/knn.so ./lib/rinha_fraud-0.1.0/priv/

ENV MIX_ENV=prod

CMD ["./bin/rinha_fraud", "start"]
