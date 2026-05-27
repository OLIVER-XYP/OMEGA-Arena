#include "MiningPhase.hpp"

#include <cmath>
#include <optional>
#include <vector>

#include "EventBuffer.hpp"
#include "LocalStepResult.hpp"
#include "TaskExecutor.hpp"

namespace hlt::rules::phases {
namespace {

struct MineEffect {
    Entity::id_type entity_id{Entity::None};
    Player::id_type owner{Player::None};
    Location location{0, 0};
    energy_type extracted{};
    energy_type gained{};
    bool was_captured{};
};

std::vector<MineEffect> build_mine_effects(const GameState &state,
                                           const StepResult &result,
                                           const GameConfig &config,
                                           TaskExecutor &executor) {
    std::vector<std::pair<Entity::id_type, std::reference_wrapper<const Entity>>> entities;
    entities.reserve(state.store_ref().entities_ref().size());
    for (const auto &[entity_id, entity] : state.store_ref().entities_ref()) {
        entities.emplace_back(entity_id, std::cref(entity));
    }

    std::vector<std::optional<MineEffect>> planned_effects(entities.size());
    const auto &store = state.store_ref();
    const auto &map = state.map_ref();
    const auto &economy = config.ruleset.economy;
    const auto &inspiration = config.ruleset.inspiration;
    const auto max_energy = economy.max_energy;
    const auto bonus_multiplier = inspiration.bonus_multiplier;

    executor.parallel_for(0, entities.size(), [&](std::size_t index) {
        const auto &[entity_id, entity_ref] = entities[index];
        const auto &entity = entity_ref.get();

        // A defending ship stayed put, so it can still mine its cell when
        // DEFEND_ALLOWS_MINING is set — this removes the economy cost of defending
        // (and the pin/freeze exploit), letting a defensive strategy hold ground.
        const bool defend_mines = config.ruleset.combat.defend_allows_mining && entity.is_defending;
        const bool already_acted = result.changed_entities.find(entity_id) != result.changed_entities.end();
        if ((already_acted && !defend_mines) || entity.energy >= max_energy) {
            return;
        }

        const auto &player = store.players_ref().at(entity.owner);
        const auto location = player.get_entity_location(entity_id);
        const auto &cell = map.at(location);
        const auto ratio = entity.is_inspired ? inspiration.extract_ratio : economy.extract_ratio;

        energy_type extracted = static_cast<energy_type>(std::ceil(static_cast<double>(cell.energy) / ratio));
        energy_type gained = extracted;

        if (extracted == 0 && cell.energy > 0) {
            extracted = gained = cell.energy;
        }

        if (extracted + entity.energy > max_energy) {
            extracted = max_energy - entity.energy;
        }

        if (entity.is_inspired && bonus_multiplier > 0) {
            gained += static_cast<energy_type>(bonus_multiplier * gained);
        }

        if (max_energy - entity.energy < gained) {
            gained = max_energy - entity.energy;
        }

        MineEffect effect{};
        effect.entity_id = entity_id;
        effect.owner = entity.owner;
        effect.location = location;
        effect.extracted = extracted;
        effect.gained = gained;
        effect.was_captured = entity.was_captured;
        planned_effects[index] = effect;
    });

    std::vector<MineEffect> effects;
    effects.reserve(planned_effects.size());
    for (auto &effect : planned_effects) {
        if (effect.has_value()) {
            effects.push_back(*effect);
        }
    }
    return effects;
}

void apply_mine_effects(GameState &state,
                        StepResult &result,
                        events::EventSink &event_sink,
                        const std::vector<MineEffect> &effects) {
    auto &store = state.store_ref();
    auto &map = state.map_ref();
    LocalStepResult local_result;
    events::EventBuffer event_buffer;

    for (const auto &effect : effects) {
        auto &entity = store.get_entity(effect.entity_id);
        auto &cell = map.at(effect.location);
        entity.energy += effect.gained;
        cell.energy -= effect.extracted;
        store.map_total_energy -= effect.extracted;
        store.changed_cells_ref().emplace(effect.location);
        local_result.changed_cells.emplace(effect.location);
        event_buffer.emit(events::MinedEvent{effect.entity_id,
                                             effect.owner,
                                             effect.location,
                                             effect.extracted,
                                             effect.gained,
                                             effect.was_captured});
    }

    merge_into(result, std::move(local_result));
    event_buffer.flush_into(event_sink);
}

} // namespace

void MiningPhase::execute(RuleContext &context) const {
    InlineExecutor fallback_executor;
    auto *executor = context.executor() != nullptr ? context.executor() : &fallback_executor;
    const auto effects = build_mine_effects(context.state(), context.result(), context.config(), *executor);
    apply_mine_effects(context.state(), context.result(), context.event_sink(), effects);
}

} // namespace hlt::rules::phases
