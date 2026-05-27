#ifndef ENTITY_H
#define ENTITY_H

#include "Constants.hpp"
#include "Enumerated.hpp"

namespace hlt {

// Forward declare to avoid circular header dependency.
struct Player;

using player_id_type = class_id<Player>;

/** A player-affiliated entity placed on the Halite map. */
struct Entity final : public Enumerated<Entity> {
    friend class Factory<Entity>;

    const player_id_type owner; /**< Owner of the entity. */
    energy_type energy;         /**< Energy of the entity. */
    int hp;                     /**< Current HP of the entity. */
    bool was_captured;          /**< Track whether this entity was captured for statistics purposes. */
    bool is_inspired;           /**< Track whether or not this entity is currently inspired. */
    bool is_defending;          /**< Set during the defend phase and cleared at the start of the next defend phase. */
    int protection_turns;       /**< Remaining turns of emergency-spawn protection; forces is_defending while > 0. */
    // Per-ship audit fields (tracked for ROI analysis; set on spawn/combat events).
    unsigned long spawn_turn;      /**< Turn this entity was created. */
    energy_type lifetime_deposited; /**< Cumulative halite this entity has deposited back to its player. */
    energy_type enemy_halite_taken; /**< Cumulative halite stolen via attacks + kill-cargo credited to this ship's owner. */
    int enemy_hp_dealt;            /**< Cumulative HP damage this ship has inflicted on enemy ships. */

    /**
     * Convert an Entity to JSON format.
     * @param[out] json The output JSON.
     * @param entity The entity to convert.
     */
    friend void to_json(nlohmann::json &json, const Entity &entity);

    /**
     * Write an Entity to bot serial format.
     * @param ostream The output stream.
     * @param entity The entity to write.
     * @return The output stream.
     */
    friend std::ostream &operator<<(std::ostream &ostream, const Entity &entity);

private:
    /**
     * Create Entity from ID, owner ID, and energy.
     * @param id The entity ID.
     * @param owner The owner ID.
     * @param energy The energy.
     */
    Entity(id_type id, player_id_type owner, energy_type energy) : Enumerated(id), owner(owner), energy(energy), hp(Constants::get().INITIAL_HP), was_captured(false), is_inspired(false), is_defending(false), protection_turns(0), spawn_turn(0), lifetime_deposited(0), enemy_halite_taken(0), enemy_hp_dealt(0) {}
};

}

#endif // ENTITY_H
