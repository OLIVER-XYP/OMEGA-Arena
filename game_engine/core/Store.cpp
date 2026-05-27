#include "Store.hpp"

namespace hlt {

/**
 * Get a player by ID.
 *
 * @param id The player ID.
 * @return The player.
 */
Player &Store::get_player(const Player::id_type &id) {
    auto iterator = players.find(id);
    assert(iterator != players.end());
    return iterator->second;
}

/**
 * Get an entity by ID.
 *
 * @param id The entity ID.
 * @return The entity.
 */
Entity &Store::get_entity(const Entity::id_type &id) {
    auto iterator = entities.find(id);
    assert(iterator != entities.end());
    return iterator->second;
}

/**
 * Get an entity by ID.
 *
 * @param id The entity ID.
 * @return The entity.
 */
const Entity &Store::get_entity(const Entity::id_type &id) const {
    auto iterator = entities.find(id);
    assert(iterator != entities.end());
    return iterator->second;
}

/**
 * Obtain a new entity.
 *
 * @param energy The energy of the entity.
 * @param owner The owner of the entity.
 * @return The new entity.
 */
Entity &Store::new_entity(energy_type energy, const Player::id_type &owner) {
    auto entity = entity_factory.make(owner, energy);
    return entities.emplace(entity.id, entity).first->second;
}

/**
 * Delete an entity by ID.
 *
 * @param id The ID of the entity.
 */
void Store::delete_entity(const Entity::id_type id) {
    auto iterator = entities.find(id);
    assert(iterator != entities.end());
    if (audit_enabled) {
        const auto &e = iterator->second;
        ship_audits.push_back(ShipAudit{
            static_cast<int>(e.owner.value),
            static_cast<long>(e.id.value),
            e.spawn_turn,
            current_turn,
            e.lifetime_deposited,
            e.enemy_halite_taken,
            e.enemy_hp_dealt,
            false  // died during play
        });
    }
    entities.erase(iterator);
}

/**
 * Obtain a new dropoff.
 * @param location The location of the dropoff.
 * @return The new dropoff.
 */
Dropoff Store::new_dropoff(Location location) {
    return dropoff_factory.make(location);
}

}
