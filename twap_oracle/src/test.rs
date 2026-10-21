#![cfg(test)]

use super::*;
use soroban_sdk::testutils::{Address as _, Ledger as _};
use soroban_sdk::Env;

struct Setup {
    env: Env,
    admin: Address,
    base: Address,
    quote: Address,
    client: TwapOracleClient<'static>,
}

fn setup() -> Setup {
    let env = Env::default();
    env.mock_all_auths();

    // Put the ledger clock ahead of every timestamp used in the tests so that
    // the "not in the future" guard does not interfere with the fixtures.
    env.ledger().with_mut(|ledger| ledger.timestamp = 1_000_000);

    let admin = Address::generate(&env);
    let base = Address::generate(&env);
    let quote = Address::generate(&env);
    let id = env.register_contract(None, TwapOracle);
    let client = TwapOracleClient::new(&env, &id);
    client.initialize(&admin);

    Setup {
        env,
        admin,
        base,
        quote,
        client,
    }
}

fn set_time(env: &Env, timestamp: u64) {
    env.ledger().with_mut(|ledger| ledger.timestamp = timestamp);
}

/// 100 for 60s, then 200 for 60s, then 300. Segment integrals:
/// obs0.cumulative = 0, obs1.cumulative = 6_000, obs2.cumulative = 18_000.
fn seed_observations(setup: &Setup) {
    set_time(&setup.env, 1_000);
    let _ = setup
        .client
        .observe(&setup.base, &setup.quote, &100, &1_000);
    set_time(&setup.env, 1_060);
    let _ = setup
        .client
        .observe(&setup.base, &setup.quote, &200, &1_060);
    set_time(&setup.env, 1_120);
    let _ = setup
        .client
        .observe(&setup.base, &setup.quote, &300, &1_120);
}

#[test]
fn initialize_sets_admin_and_empty_history() {
    let setup = setup();

    assert_eq!(setup.client.admin(), Some(setup.admin.clone()));
    assert_eq!(
        setup.client.observation_count(&setup.base, &setup.quote),
        0
    );
    assert_eq!(
        setup.client.latest_observation(&setup.base, &setup.quote),
        None
    );
}

#[test]
fn initialize_cannot_be_called_twice() {
    let setup = setup();

    assert_eq!(
        setup.client.try_initialize(&setup.admin),
        Err(Ok(Error::AlreadyInitialized))
    );
}

#[test]
fn observe_requires_initialization() {
    let env = Env::default();
    env.mock_all_auths();
    env.ledger().with_mut(|ledger| ledger.timestamp = 1_000);

    let base = Address::generate(&env);
    let quote = Address::generate(&env);
    let id = env.register_contract(None, TwapOracle);
    let client = TwapOracleClient::new(&env, &id);

    assert_eq!(
        client.try_observe(&base, &quote, &100, &1_000),
        Err(Ok(Error::NotInitialized))
    );
}

#[test]
fn observe_appends_and_derives_cumulative_price() {
    let setup = setup();
    seed_observations(&setup);

    let observations = setup.client.observations(&setup.base, &setup.quote);
    assert_eq!(observations.len(), 3);
    assert_eq!(
        setup.client.observation_count(&setup.base, &setup.quote),
        3
    );

    let first = observations.get(0).unwrap();
    assert_eq!(first.timestamp, 1_000);
    assert_eq!(first.price, 100);
    assert_eq!(first.cumulative_price, u256::ZERO);

    let second = observations.get(1).unwrap();
    assert_eq!(second.cumulative_price, u256::from(6_000u64));

    let third = observations.get(2).unwrap();
    assert_eq!(third.cumulative_price, u256::from(18_000u64));

    let latest = setup
        .client
        .latest_observation(&setup.base, &setup.quote)
        .unwrap();
    assert_eq!(latest, third);
}

#[test]
fn observe_rejects_non_positive_and_out_of_range_prices() {
    let setup = setup();

    assert_eq!(
        setup
            .client
            .try_observe(&setup.base, &setup.quote, &0, &1_000),
        Err(Ok(Error::InvalidPrice))
    );
    assert_eq!(
        setup
            .client
            .try_observe(&setup.base, &setup.quote, &-1, &1_000),
        Err(Ok(Error::InvalidPrice))
    );
    assert_eq!(
        setup
            .client
            .try_observe(&setup.base, &setup.quote, &(MAX_PRICE + 1), &1_000),
        Err(Ok(Error::InvalidPrice))
    );
    // The upper bound itself is accepted.
    assert_eq!(
        setup
            .client
            .observe(&setup.base, &setup.quote, &MAX_PRICE, &1_000),
        1
    );
}

#[test]
fn observe_rejects_future_timestamps() {
    let setup = setup();

    set_time(&setup.env, 1_000);
    assert_eq!(
        setup
            .client
            .try_observe(&setup.base, &setup.quote, &100, &1_001),
        Err(Ok(Error::TimestampInFuture))
    );
}

#[test]
fn observe_rejects_non_monotonic_timestamps() {
    let setup = setup();
    seed_observations(&setup);

    // Equal to the previous timestamp.
    assert_eq!(
        setup
            .client
            .try_observe(&setup.base, &setup.quote, &150, &1_120),
        Err(Ok(Error::TimestampNotMonotonic))
    );
    // Older than the previous timestamp.
    assert_eq!(
        setup
            .client
            .try_observe(&setup.base, &setup.quote, &150, &1_100),
        Err(Ok(Error::TimestampNotMonotonic))
    );
}

#[test]
fn observe_enforces_the_minimum_interval() {
    let setup = setup();
    set_time(&setup.env, 1_000);
    let _ = setup
        .client
        .observe(&setup.base, &setup.quote, &100, &1_000);

    set_time(&setup.env, 1_059);
    assert_eq!(
        setup
            .client
            .try_observe(&setup.base, &setup.quote, &200, &1_059),
        Err(Ok(Error::ObservationTooSoon))
    );

    // Exactly MIN_INTERVAL_SECONDS later is accepted.
    set_time(&setup.env, 1_060);
    assert_eq!(
        setup
            .client
            .observe(&setup.base, &setup.quote, &200, &1_060),
        2
    );
}

#[test]
fn observe_prunes_the_oldest_observation_when_full() {
    let setup = setup();
    set_time(&setup.env, 1_000);

    let mut index = 0u64;
    while index < u64::from(MAX_OBSERVATIONS) + 1 {
        let timestamp = 1_000 + index * MIN_INTERVAL_SECONDS;
        set_time(&setup.env, timestamp);
        let _ = setup
            .client
            .observe(&setup.base, &setup.quote, &100, &timestamp);
        index += 1;
    }

    assert_eq!(
        setup.client.observation_count(&setup.base, &setup.quote),
        MAX_OBSERVATIONS
    );

    let observations = setup.client.observations(&setup.base, &setup.quote);
    assert_eq!(observations.len(), MAX_OBSERVATIONS);
    // The very first observation was dropped.
    assert_eq!(
        observations.get(0).unwrap().timestamp,
        1_000 + MIN_INTERVAL_SECONDS
    );
}

#[test]
fn observe_handles_high_price_elapsed_product_without_overflow() {
    let setup = setup();

    // Advance the ledger clock by an astronomically large gap with MAX_PRICE.
    // MAX_PRICE * elapsed exceeds i128::MAX, but u256 accumulator handles it seamlessly without panic.
    set_time(&setup.env, 1_000);
    let _ = setup
        .client
        .observe(&setup.base, &setup.quote, &MAX_PRICE, &1_000);

    set_time(&setup.env, 1_000 + 10_000_000);
    let count = setup
        .client
        .observe(&setup.base, &setup.quote, &MAX_PRICE, &(1_000 + 10_000_000));
    assert_eq!(count, 2);

    let latest = setup
        .client
        .latest_observation(&setup.base, &setup.quote)
        .unwrap();
    let expected_cum = u256::from(MAX_PRICE as u128) * u256::from(10_000_000u64);
    assert_eq!(latest.cumulative_price, expected_cum);

    // TWAP calculation over a window within this gap returns MAX_PRICE exactly.
    assert_eq!(
        setup.client.twap(&setup.base, &setup.quote, &60),
        MAX_PRICE
    );
}

#[test]
fn twap_calculates_correctly_across_accumulator_wrap_boundary() {
    let setup = setup();

    // Directly seed storage with an observation whose cumulative_price is near u256::MAX.
    // Near 2^256 - 1:
    let base_cum = u256::MAX - u256::from(5_000u64);
    let mut initial_obs = Vec::new(&setup.env);
    initial_obs.push_back(Observation {
        timestamp: 1_000,
        price: 100,
        cumulative_price: base_cum,
    });
    setup.env.storage().instance().set(
        &DataKey::Pair(setup.base.clone(), setup.quote.clone()),
        &initial_obs,
    );

    // After 100 seconds at price 100, elapsed * price = 10_000.
    // Adding 10_000 to (u256::MAX - 5_000) wraps around modulo 2^256:
    // (u256::MAX - 5_000) + 10_000 = 4_999 (as u256::MAX + 1 = 0).
    set_time(&setup.env, 1_100);
    let count = setup
        .client
        .observe(&setup.base, &setup.quote, &200, &1_100);
    assert_eq!(count, 2);

    let latest = setup
        .client
        .latest_observation(&setup.base, &setup.quote)
        .unwrap();
    let expected_wrapped_cum = u256::from(4_999u64);
    assert_eq!(latest.cumulative_price, expected_wrapped_cum);
    assert!(latest.cumulative_price < base_cum);

    // TWAP over the full 100s window across the wrap boundary:
    // (4_999 - (u256::MAX - 5_000)) mod 2^256 = 10_000.
    // 10_000 / 100 = 100.
    assert_eq!(setup.client.twap(&setup.base, &setup.quote, &100), 100);

    // TWAP over a 50s window with interpolation across the wrap boundary:
    // Target timestamp = 1_100 - 50 = 1_050.
    // Interpolated cumulative at target = base_cum + 50 * 100 = base_cum + 5_000 = u256::MAX.
    // Difference = 4_999.wrapping_sub(u256::MAX) = 5_000.
    // 5_000 / 50 = 100.
    assert_eq!(setup.client.twap(&setup.base, &setup.quote, &50), 100);
}


#[test]
fn twap_averages_over_the_requested_window() {
    let setup = setup();
    seed_observations(&setup);

    // The full 120s of history: (100*60 + 200*60) / 120.
    assert_eq!(setup.client.twap(&setup.base, &setup.quote, &120), 150);
    // Only the last segment, exactly one observation wide.
    assert_eq!(setup.client.twap(&setup.base, &setup.quote, &60), 200);
}

#[test]
fn twap_interpolates_the_window_start() {
    let setup = setup();
    seed_observations(&setup);

    // Window starts inside the first segment: (100*30 + 200*60) / 90.
    assert_eq!(setup.client.twap(&setup.base, &setup.quote, &90), 166);
    // Window starts inside the second segment: (200*30 + 300*0) / 30.
    assert_eq!(setup.client.twap(&setup.base, &setup.quote, &30), 200);
}

#[test]
fn twap_validates_the_window_bounds() {
    let setup = setup();
    seed_observations(&setup);

    assert_eq!(
        setup.client.try_twap(&setup.base, &setup.quote, &0),
        Err(Ok(Error::InvalidWindow))
    );
    assert_eq!(
        setup
            .client
            .try_twap(&setup.base, &setup.quote, &(MAX_WINDOW_SECONDS + 1)),
        Err(Ok(Error::InvalidWindow))
    );
}

#[test]
fn twap_requires_enough_retained_history() {
    let setup = setup();
    seed_observations(&setup);

    // 200s back reaches before the first observation (1_000).
    assert_eq!(
        setup.client.try_twap(&setup.base, &setup.quote, &200),
        Err(Ok(Error::InsufficientHistory))
    );
}

#[test]
fn twap_without_observations_reports_no_data() {
    let setup = setup();

    assert_eq!(
        setup.client.try_twap(&setup.base, &setup.quote, &60),
        Err(Ok(Error::NoObservations))
    );
}

#[test]
fn twap_rejects_a_window_wider_than_history_with_one_observation() {
    let setup = setup();
    set_time(&setup.env, 1_000);
    let _ = setup
        .client
        .observe(&setup.base, &setup.quote, &100, &1_000);

    // A window wider than the available history is rejected rather than
    // returning the single spot price.
    assert_eq!(
        setup.client.try_twap(&setup.base, &setup.quote, &60),
        Err(Ok(Error::InsufficientHistory))
    );
}

// ── Consensus & Secondary Oracle Fallback Tests ───────────────────────────────

#[allow(dead_code)]
struct DualOracleSetup {
    env: Env,
    admin: Address,
    client: TwapOracleClient<'static>,
    secondary_admin: Address,
    secondary_client: TwapOracleClient<'static>,
    secondary_id: Address,
}

fn setup_dual_oracle() -> DualOracleSetup {
    let env = Env::default();
    env.mock_all_auths();

    env.ledger().with_mut(|ledger| ledger.timestamp = 1_000_000);

    let admin = Address::generate(&env);
    let id = env.register_contract(None, TwapOracle);
    let client = TwapOracleClient::new(&env, &id);
    client.initialize(&admin);

    let secondary_admin = Address::generate(&env);
    let secondary_id = env.register_contract(None, TwapOracle);
    let secondary_client = TwapOracleClient::new(&env, &secondary_id);
    secondary_client.initialize(&secondary_admin);

    client.set_secondary_oracle(&secondary_id);

    DualOracleSetup {
        env,
        admin,
        client,
        secondary_admin,
        secondary_client,
        secondary_id,
    }
}

#[test]
fn secondary_oracle_registration_and_persistence() {
    let setup = setup();
    assert_eq!(setup.client.secondary_oracle(), None);

    let secondary_id = Address::generate(&setup.env);
    setup.client.set_secondary_oracle(&secondary_id);
    assert_eq!(setup.client.secondary_oracle(), Some(secondary_id.clone()));

    // Test register alias
    let another_id = Address::generate(&setup.env);
    setup.client.register_secondary_oracle(&another_id);
    assert_eq!(setup.client.secondary_oracle(), Some(another_id));
}

#[test]
fn configuration_setters_and_getters() {
    let setup = setup();

    // Default max_divergence_bps is 500
    assert_eq!(setup.client.max_divergence_bps(), 500);
    setup.client.set_max_divergence_bps(&300);
    assert_eq!(setup.client.max_divergence_bps(), 300);
    assert_eq!(
        setup.client.try_set_max_divergence_bps(&10_001),
        Err(Ok(Error::InvalidConfig))
    );

    // Default staleness_threshold is 3600
    assert_eq!(setup.client.staleness_threshold(), 3_600);
    setup.client.set_staleness_threshold(&1_800);
    assert_eq!(setup.client.staleness_threshold(), 1_800);
    assert_eq!(
        setup.client.try_set_staleness_threshold(&0),
        Err(Ok(Error::InvalidConfig))
    );

    // Default consensus_weights is (5000, 5000)
    assert_eq!(setup.client.consensus_weights(), (5_000, 5_000));
    setup.client.set_consensus_weights(&6_000, &4_000);
    assert_eq!(setup.client.consensus_weights(), (6_000, 4_000));
    assert_eq!(
        setup.client.try_set_consensus_weights(&0, &5_000),
        Err(Ok(Error::InvalidConfig))
    );
    assert_eq!(
        setup.client.try_set_consensus_weights(&5_000, &0),
        Err(Ok(Error::InvalidConfig))
    );
}

#[test]
fn consult_returns_weighted_average_when_both_fresh_and_within_threshold() {
    let setup = setup_dual_oracle();

    // Primary: 100 at 1000, 100 at 1060, 100 at 1120 -> twap(60) = 100
    set_time(&setup.env, 1_000);
    let _ = setup.client.observe(&100, &1_000);
    set_time(&setup.env, 1_060);
    let _ = setup.client.observe(&100, &1_060);
    set_time(&setup.env, 1_120);
    let _ = setup.client.observe(&100, &1_120);

    // Secondary: 104 at 1000, 104 at 1060, 104 at 1120 -> twap(60) = 104
    let _ = setup.secondary_client.observe(&104, &1_000);
    let _ = setup.secondary_client.observe(&104, &1_060);
    let _ = setup.secondary_client.observe(&104, &1_120);

    // Ledger clock is 1120 (fresh for both: elapsed = 0)
    // Divergence: (104 - 100) * 10_000 / 100 = 400 bps <= 500 bps (max_divergence_bps)
    // Consensus weighted average (50/50): (100 + 104) / 2 = 102
    assert_eq!(setup.client.consult(&60), 102);
}

#[test]
fn consult_respects_custom_consensus_weights() {
    let setup = setup_dual_oracle();

    set_time(&setup.env, 1_000);
    let _ = setup.client.observe(&100, &1_000);
    let _ = setup.secondary_client.observe(&104, &1_000);

    set_time(&setup.env, 1_060);
    let _ = setup.client.observe(&100, &1_060);
    let _ = setup.secondary_client.observe(&104, &1_060);

    set_time(&setup.env, 1_120);
    let _ = setup.client.observe(&100, &1_120);
    let _ = setup.secondary_client.observe(&104, &1_120);

    // 75% primary, 25% secondary: (100 * 7500 + 104 * 2500) / 10000 = 101
    setup.client.set_consensus_weights(&7_500, &2_500);
    assert_eq!(setup.client.consult(&60), 101);
}

#[test]
fn consult_trips_circuit_breaker_when_discrepancy_exceeds_threshold() {
    let setup = setup_dual_oracle();

    // Primary: 100
    set_time(&setup.env, 1_000);
    let _ = setup.client.observe(&100, &1_000);
    set_time(&setup.env, 1_060);
    let _ = setup.client.observe(&100, &1_060);
    set_time(&setup.env, 1_120);
    let _ = setup.client.observe(&100, &1_120);

    // Secondary: 106 -> Divergence = (106 - 100) * 10_000 / 100 = 600 bps > 500 bps
    let _ = setup.secondary_client.observe(&106, &1_000);
    let _ = setup.secondary_client.observe(&106, &1_060);
    let _ = setup.secondary_client.observe(&106, &1_120);

    assert!(!setup.client.is_circuit_breaker_tripped());

    // Should trip the circuit breaker and revert
    assert_eq!(
        setup.client.try_consult(&60),
        Err(Ok(Error::CircuitBreakerTripped))
    );

    // In Soroban, invocations returning Err revert uncommitted state.
    // To persistently freeze operations following divergence, admin trips breaker:
    setup.client.trip_circuit_breaker();

    // Circuit breaker is now persistently tripped
    assert!(setup.client.is_circuit_breaker_tripped());

    // Subsequent calls immediately fail with CircuitBreakerTripped
    assert_eq!(
        setup.client.try_consult(&60),
        Err(Ok(Error::CircuitBreakerTripped))
    );
}

#[test]
fn circuit_breaker_manual_trip_and_reset() {
    let setup = setup_dual_oracle();

    assert!(!setup.client.is_circuit_breaker_tripped());

    // Admin can manually trip the circuit breaker
    setup.client.trip_circuit_breaker();
    assert!(setup.client.is_circuit_breaker_tripped());

    // Calls to consult revert while tripped
    assert_eq!(
        setup.client.try_consult(&60),
        Err(Ok(Error::CircuitBreakerTripped))
    );

    // Admin can reset the circuit breaker
    setup.client.reset_circuit_breaker();
    assert!(!setup.client.is_circuit_breaker_tripped());
}

#[test]
fn consult_falls_back_to_secondary_when_primary_is_stale() {
    let setup = setup_dual_oracle();

    // Primary observations are written at t = 1000..1120
    set_time(&setup.env, 1_000);
    let _ = setup.client.observe(&100, &1_000);
    set_time(&setup.env, 1_060);
    let _ = setup.client.observe(&100, &1_060);
    set_time(&setup.env, 1_120);
    let _ = setup.client.observe(&100, &1_120);

    // Secondary observations are written fresh at t = 5000..5120
    set_time(&setup.env, 5_000);
    let _ = setup.secondary_client.observe(&250, &5_000);
    set_time(&setup.env, 5_060);
    let _ = setup.secondary_client.observe(&250, &5_060);
    set_time(&setup.env, 5_120);
    let _ = setup.secondary_client.observe(&250, &5_120);

    // Current ledger time is 5120
    // Primary elapsed = 5120 - 1120 = 4000s > staleness_threshold (3600s) -> STALE
    // Secondary elapsed = 5120 - 5120 = 0s <= 3600s -> FRESH
    // Should fallback to secondary price (250)
    assert_eq!(setup.client.consult(&60), 250);
}

#[test]
fn consult_fails_when_both_primary_and_secondary_are_stale() {
    let setup = setup_dual_oracle();

    // Primary written at t = 1000..1120
    set_time(&setup.env, 1_000);
    let _ = setup.client.observe(&100, &1_000);
    set_time(&setup.env, 1_060);
    let _ = setup.client.observe(&100, &1_060);
    set_time(&setup.env, 1_120);
    let _ = setup.client.observe(&100, &1_120);

    // Secondary also written at t = 1000..1120
    let _ = setup.secondary_client.observe(&100, &1_000);
    let _ = setup.secondary_client.observe(&100, &1_060);
    let _ = setup.secondary_client.observe(&100, &1_120);

    // Advance ledger clock to 10000 (> 3600s elapsed for both)
    set_time(&setup.env, 10_000);

    assert_eq!(setup.client.try_consult(&60), Err(Ok(Error::PrimaryStale)));
}

#[test]
fn consult_without_secondary_uses_primary_if_fresh() {
    let setup = setup();
    seed_observations(&setup);

    // Ledger clock at 1120 (same as latest observation)
    set_time(&setup.env, 1_120);
    assert_eq!(setup.client.consult(&60), 200);
}

#[test]
fn consult_without_secondary_fails_if_primary_stale() {
    let setup = setup();
    seed_observations(&setup);

    // Advance clock past staleness limit
    set_time(&setup.env, 10_000);
    assert_eq!(setup.client.try_consult(&60), Err(Ok(Error::PrimaryStale)));
}

#[test]
fn consult_with_zero_window_returns_latest_spot_consensus() {
    let setup = setup_dual_oracle();

    set_time(&setup.env, 1_000);
    let _ = setup.client.observe(&300, &1_000);
    let _ = setup.secondary_client.observe(&310, &1_000);

    // Divergence: (310 - 300) * 10_000 / 300 = 333 bps <= 500 bps
    // Spot consensus: (300 + 310) / 2 = 305
    assert_eq!(setup.client.consult(&0), 305);
}
