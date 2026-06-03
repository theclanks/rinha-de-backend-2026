math.randomseed(os.time())

request = function()
  r = math.random()
  -- 60% casos fáceis (LDA -> Legit)
  if r < 0.6 then
    return wrk.format("POST", "/fraud-score", {["Content-Type"] = "application/json"}, '{"id":"1","transaction":{"amount":41.12,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},"customer":{"avg_amount":82.24,"tx_count_24h":3,"known_merchants":["MERC-003","MERC-016"]},"merchant":{"id":"MERC-016","mcc":"5411","avg_amount":60.25},"terminal":{"is_online":false,"card_present":true,"km_from_home":29.23},"last_transaction":null}')
  -- 30% casos de fraude (LDA -> Fraud)
  elseif r < 0.9 then
    return wrk.format("POST", "/fraud-score", {["Content-Type"] = "application/json"}, '{"id":"2","transaction":{"amount":9505.97,"installments":10,"requested_at":"2026-03-14T05:15:12Z"},"customer":{"avg_amount":81.28,"tx_count_24h":20,"known_merchants":["MERC-008","MERC-007","MERC-005"]},"merchant":{"id":"MERC-068","mcc":"7802","avg_amount":54.86},"terminal":{"is_online":false,"card_present":true,"km_from_home":952.27},"last_transaction":null}')
  -- 10% casos difíceis (KNN)
  else
    return wrk.format("POST", "/fraud-score", {["Content-Type"] = "application/json"}, '{"id":"3","transaction":{"amount":3000,"installments":5,"requested_at":"2026-03-14T05:15:12Z"},"customer":{"avg_amount":400,"tx_count_24h":8,"known_merchants":["MERC-001","MERC-002"]},"merchant":{"id":"MERC-050","mcc":"5912","avg_amount":150},"terminal":{"is_online":false,"card_present":true,"km_from_home":300},"last_transaction":{"timestamp":"2026-03-13T20:00:00Z","km_from_current":50}}')
  end
end
