import * as PIXI from "pixi.js";

import * as assets from "./assets";
import {CELL_SIZE, PLAYER_COLORS} from "./assets";
import theme from "./theme";

const themeParams = theme();

/**
 * Draws a geometric action badge into a PIXI.Graphics at local coords (0,0).
 * s  = half-size of the icon in pixels (scales with camera zoom)
 * command = { type, direction } or null
 */
function drawActionIcon(g, command, s) {
    g.clear();
    if (!command) return;

    const type = command.type;
    const dir  = command.direction;
    const isMine = type === "m" && (!dir || dir === "o");
    const isMove = type === "m" && dir && dir !== "o";

    if (isMine) {
        // ⊕  Gold sparkle — 8 alternating long/short rays
        const lw = Math.max(1, s * 0.16);
        g.lineStyle(lw, 0xFFD700, 0.95);
        for (let i = 0; i < 8; i++) {
            const angle  = (i / 8) * Math.PI * 2;
            const outerR = s * (i % 2 === 0 ? 0.92 : 0.58);
            const innerR = s * 0.22;
            g.moveTo(Math.cos(angle) * innerR, Math.sin(angle) * innerR);
            g.lineTo(Math.cos(angle) * outerR, Math.sin(angle) * outerR);
        }
        g.lineStyle(0);
        g.beginFill(0xFFD700, 0.9);
        g.drawCircle(0, 0, s * 0.24);
        g.endFill();

    } else if (isMove) {
        // ▶  Light-gray filled arrowhead pointing in movement direction
        const ang = { n: -Math.PI / 2, s: Math.PI / 2, e: 0, w: Math.PI }[dir] || 0;
        const cos = Math.cos(ang), sin = Math.sin(ang);
        // perpendicular unit vector
        const px = -sin, py = cos;
        g.lineStyle(0);
        g.beginFill(0xDDDDDD, 0.85);
        g.drawPolygon([
            cos * s * 0.88,  sin * s * 0.88,
            (cos * -0.1 + px * 0.68) * s, (sin * -0.1 + py * 0.68) * s,
            (cos * -0.1 - px * 0.68) * s, (sin * -0.1 - py * 0.68) * s,
        ]);
        g.endFill();

    } else if (type === "a") {
        // ✕  Red bold cross (attack / strike)
        const lw = Math.max(1.5, s * 0.30);
        g.lineStyle(lw, 0xFF1133, 0.97);
        g.moveTo(-s * 0.72, -s * 0.72);
        g.lineTo( s * 0.72,  s * 0.72);
        g.moveTo( s * 0.72, -s * 0.72);
        g.lineTo(-s * 0.72,  s * 0.72);
        // center dot
        g.lineStyle(0);
        g.beginFill(0xFF4455, 0.9);
        g.drawCircle(0, 0, s * 0.22);
        g.endFill();

    } else if (type === "d") {
        // ◑  Blue shield arc — partial circle open at bottom
        const lw = Math.max(1.5, s * 0.24);
        g.lineStyle(lw, 0x44AAFF, 0.92);
        // Arc from −150° to +150° (top & sides, open at bottom)
        g.arc(0, 0, s * 0.80, -Math.PI * 5 / 6, Math.PI * 5 / 6, false);
        // Close the shield bottom
        const bx = Math.cos(Math.PI * 5 / 6) * s * 0.80;
        const by = Math.sin(Math.PI * 5 / 6) * s * 0.80;
        g.lineTo(0, s * 0.30);
        g.lineTo(-bx, by);

    } else if (type === "h") {
        // ✚  Green thick plus (heal)
        const arm = s * 0.82, thk = s * 0.30;
        g.lineStyle(0);
        g.beginFill(0x33FF77, 0.92);
        g.drawRect(-thk, -arm,  thk * 2, arm * 2);
        g.drawRect(-arm, -thk,  arm * 2, thk * 2);
        g.endFill();

    } else if (type === "g") {
        // ○  Cyan ring (spawned this turn)
        g.lineStyle(Math.max(1, s * 0.20), 0x66FFEE, 0.80);
        g.drawCircle(0, 0, s * 0.72);
        g.lineStyle(0);
        g.beginFill(0x66FFEE, 0.55);
        g.drawCircle(0, 0, s * 0.26);
        g.endFill();
    }
}

/**
 * Manages a ship on screen.
 */
export default class Ship {
    /**
     * @param visualizer The visualizer object
     * @param record The sprite record. {x, y, owner, energy}
     */
    constructor(visualizer, record) {
        const spriteShape = new PIXI.Graphics();
        spriteShape.beginFill(assets.SPRITE_COLOR, 1);
        spriteShape.drawCircle(0, 0, assets.CELL_SIZE * 64);
        spriteShape.endFill();

        let spriteTexture = visualizer.application.renderer.generateTexture(spriteShape);

        this.sprite = new PIXI.extras.AnimatedSprite(assets.TURTLE_SPRITES[record.owner]);
        this.inspiredSprite = new PIXI.Sprite(spriteTexture);
        this.inspiredSprite.tint = assets.PLAYER_COLORS[record.owner];
        this.highlight = new PIXI.Sprite(spriteTexture);
        this.highlight.visible = false;
        this.highlight.alpha = 0.7;

        this.halo = new PIXI.Sprite(assets.HALO_SPRITE);
        this.halo.visible = false;

        // Action icon: geometric badge above the ship
        this.actionIcon = new PIXI.Graphics();
        this._lastActionKey = null;

        // HP bar: colored bar below the ship sprite
        this.hpBar = new PIXI.Graphics();
        this._lastHpKey = null;

        this.container = null;
        this.visualizer = visualizer;

        this.map_width  = this.visualizer.replay.production_map.width;
        this.map_height = this.visualizer.replay.production_map.height;

        this.owner  = record.owner;
        this.energy = record.energy;
        this.id     = record.id;
        this.x      = record.x;
        this.y      = record.y;

        let setupSprite = (sprite, width) => {
            sprite.width = sprite.height = width;
            sprite.anchor.x = sprite.anchor.y = 0.5;
        };

        const width = 1.1 * assets.CELL_SIZE * this.visualizer.camera.scale;
        setupSprite(this.sprite,         width * 4 * themeParams.scale.ship);
        setupSprite(this.inspiredSprite, width * 4 * themeParams.scale.ship);
        this.inspiredSprite.visible = false;
        setupSprite(this.highlight, width * 1.25);
        setupSprite(this.halo,      width * 1.25);

        if (themeParams.tintShip) {
            this.sprite.tint = PLAYER_COLORS[this.owner];
        }

        const pixelX = this.visualizer.camera.scale * CELL_SIZE * this.x
                     + this.visualizer.camera.scale * CELL_SIZE / 2;
        const pixelY = this.visualizer.camera.scale * CELL_SIZE * this.y
                     + this.visualizer.camera.scale * CELL_SIZE / 2;
        this.sprite.position.x = pixelX;
        this.sprite.position.y = pixelY;
    }

    attach(container) {
        container.addChild(this.highlight);
        container.addChild(this.inspiredSprite);
        container.addChild(this.halo);
        container.addChild(this.hpBar);
        container.addChild(this.actionIcon);
        container.addChild(this.sprite);
        this.container = container;
    }

    destroy() {
        this.container.removeChild(this.inspiredSprite);
        this.container.removeChild(this.sprite);
        this.container.removeChild(this.halo);
        this.container.removeChild(this.highlight);
        this.container.removeChild(this.actionIcon);
        this.container.removeChild(this.hpBar);
        delete this.container;
    }

    onClick() {
        this.visualizer.onSelect("ship", {
            owner: this.owner,
            id: this.id,
        });
    }

    update(command) {
        let x_move = 0;
        let y_move = 0;

        if (this.visualizer.frame < this.visualizer.replay.full_frames.length) {
            if (command && command.type === "g") {
                return;
            }
            const ownerEntities = this.visualizer.replay
                  .full_frames[this.visualizer.frame]
                  .entities[this.owner];
            if (!ownerEntities) return;
            const entity_record = ownerEntities[this.id];
            if (!entity_record) return;

            this.energy = entity_record.energy;

            if (command && command.type === "m") {
                let direction = this.sprite.rotation;
                if (command.direction === "n") {
                    direction = 0;
                    x_move = 0; y_move = -1;
                } else if (command.direction === "e") {
                    direction = Math.PI / 2;
                    x_move = 1; y_move = 0;
                } else if (command.direction === "s") {
                    direction = Math.PI;
                    x_move = 0; y_move = 1;
                } else if (command.direction === "w") {
                    direction = -Math.PI / 2;
                    x_move = -1; y_move = 0;
                }
                if (themeParams.rotateShip) {
                    this.sprite.rotation = direction;
                }

                if (this.visualizer.frame < this.visualizer.replay.full_frames.length - 1) {
                    const next_frame = this.visualizer.replay
                          .full_frames[this.visualizer.frame + 1];
                    if (next_frame.entities[this.owner] &&
                        next_frame.entities[this.owner][this.id]) {
                        const next_record = next_frame.entities[this.owner][this.id];
                        x_move = next_record.x - entity_record.x;
                        y_move = next_record.y - entity_record.y;
                        if (x_move > 1)  x_move = -1;
                        else if (x_move < -1) x_move = 1;
                        if (y_move > 1)  y_move = -1;
                        else if (y_move < -1) y_move = 1;
                    }
                } else if (this.energy < this.visualizer.findCurrentProduction(
                        this.visualizer.frame, entity_record.x, entity_record.y)
                        / this.visualizer.replay.GAME_CONSTANTS.MOVE_COST_RATIO) {
                    x_move = y_move = 0;
                }
            }

            let t = this.visualizer.time;
            t /= 0.5;
            if (t < 1) {
                t = t * t * t / 2;
            } else {
                t -= 2;
                t = (t * t * t + 2) / 2;
            }

            this.x = (entity_record.x + x_move * t + this.map_width)  % this.map_width;
            this.y = (entity_record.y + y_move * t + this.map_height) % this.map_height;
        }
    }

    draw() {
        const size    = this.visualizer.camera.scale * CELL_SIZE;
        const [ cellX, cellY ] = this.visualizer.camera.worldToCamera(this.x, this.y);
        const pixelX  = size * cellX + this.visualizer.camera.scale * CELL_SIZE / 2;
        const pixelY  = size * cellY + this.visualizer.camera.scale * CELL_SIZE / 2;

        this.sprite.position.x = pixelX;
        this.sprite.position.y = pixelY;
        this.sprite.width = this.sprite.height = size * 1.5 * themeParams.scale.ship;

        this.inspiredSprite.position.x = pixelX;
        this.inspiredSprite.position.y = pixelY;
        this.inspiredSprite.width = this.inspiredSprite.height = size * 1.25 * themeParams.scale.ship;

        this.highlight.position.x = pixelX;
        this.highlight.position.y = pixelY;
        this.highlight.width = this.highlight.height = 0.9 * size;

        this.halo.position.x = pixelX;
        this.halo.position.y = pixelY;
        this.halo.width = this.halo.height =
            (1 + 0.25 * Math.sin(this.visualizer.time * Math.PI)) * size;

        const camera = this.visualizer.camera;
        this.highlight.visible =
            camera.selected &&
            camera.selected.type  === "ship" &&
            camera.selected.id    === this.id &&
            camera.selected.owner === this.owner;

        if (!this.visualizer.currentFrame ||
            !this.visualizer.currentFrame.entities ||
            !this.visualizer.currentFrame.entities[this.owner] ||
            !this.visualizer.currentFrame.entities[this.owner][this.id]) {
            this.actionIcon.visible = false;
            return;
        }

        const spriteRecord = this.visualizer.currentFrame.entities[this.owner][this.id];
        const maxEnergy    = this.visualizer.replay.GAME_CONSTANTS.MAX_ENERGY;
        const energyPct    = spriteRecord.energy / maxEnergy;

        if (energyPct < 0.25)      this.sprite.gotoAndStop(0);
        else if (energyPct < 0.75) this.sprite.gotoAndStop(1);
        else                        this.sprite.gotoAndStop(2);

        this.halo.visible    = spriteRecord.is_inspired;
        this.actionIcon.visible = true;

        // Resolve current command for this ship
        const ownerCmds = this.visualizer.current_commands[this.owner];
        const command   = ownerCmds ? ownerCmds[this.id] : null;

        // Half-size of the icon in pixels; icon sits above the ship body
        const iconHalfSize = size * 0.30;
        const iconKey = `${command ? command.type : "null"}-${command ? command.direction : "null"}-${iconHalfSize.toFixed(1)}`;

        if (iconKey !== this._lastActionKey) {
            this._lastActionKey = iconKey;
            drawActionIcon(this.actionIcon, command, iconHalfSize);
        }

        // Position icon centred above the ship sprite
        this.actionIcon.position.x = pixelX;
        this.actionIcon.position.y = pixelY - size * 0.75;

        // HP bar: drawn below ship, color-coded by health percentage
        const maxHp = (this.visualizer.replay.GAME_CONSTANTS.INITIAL_HP) || 100;
        const hpPct = Math.max(0, Math.min(1, spriteRecord.hp / maxHp));
        const hpKey = `${spriteRecord.hp}-${size.toFixed(1)}`;
        if (hpKey !== this._lastHpKey) {
            this._lastHpKey = hpKey;
            const barW = size * 0.80;
            const barH = size * 0.13;
            this.hpBar.clear();
            this.hpBar.beginFill(0x222222, 0.75);
            this.hpBar.drawRoundedRect(-barW / 2, 0, barW, barH, barH * 0.3);
            this.hpBar.endFill();
            const hpColor = hpPct > 0.5 ? 0x44EE44 : (hpPct > 0.25 ? 0xFFAA00 : 0xFF2233);
            this.hpBar.beginFill(hpColor, 0.92);
            this.hpBar.drawRoundedRect(-barW / 2, 0, barW * hpPct, barH, barH * 0.3);
            this.hpBar.endFill();
        }
        this.hpBar.position.x = pixelX;
        this.hpBar.position.y = pixelY + size * 0.62;
        this.hpBar.visible = true;
    }
}
