import * as assets from "./assets";
import * as PIXI from "pixi.js";
import {ShockwaveFilter} from "@pixi/filter-shockwave";

export class FrameAnimation {
    constructor(start, duration, update, draw, finish) {
        this.start = start;
        this.duration = duration;
        this.update = update;
        this.draw = draw;
        this.finish = finish;
    }
}

export class SpawnAnimation extends FrameAnimation {
    constructor({ event, frame, duration, cellSize, container, reverse=false }) {
        const filter = new ShockwaveFilter();
        filter.radius = 30;
        filter.amplitude = 5;
        filter.wavelength = 7.5;
        filter.brightness = 1.25;
        if (!container.filters) {
            container.filters = [filter];
        }
        else {
            container.filters = container.filters.concat([filter]);
        }

        super(frame, duration, () => {
        }, (camera, frameTime) => {
            // Cubic ease in out
            let t = 1 - ((duration - frameTime) / duration);
            t = - t * (t - 2);
            filter.time = t;
            const [ x, y ] = camera.worldToCamera(event.location.x + 0.5, event.location.y + 0.5);
            filter.center = [ x * cellSize * camera.scale,
                              y * cellSize * camera.scale ];
        }, () => {
            container.filters = container.filters.filter(x => x !== filter);
        });
    }
}

export class AttackBeamAnimation extends FrameAnimation {
    constructor({ event, frame, duration, container }) {
        const beam  = new PIXI.Graphics();
        const flash = new PIXI.Graphics();
        container.addChild(beam);
        container.addChild(flash);

        super(frame, duration, () => {}, (camera, elapsed) => {
            const t = elapsed / duration; // 0 (start) → 1 (end)
            const alpha = 1 - t;

            const s = camera.scale;
            const [sx, sy] = camera.worldToCamera(event.location.x + 0.5, event.location.y + 0.5);
            const [tx, ty] = camera.worldToCamera(event.target_location.x + 0.5, event.target_location.y + 0.5);
            const x1 = sx * s, y1 = sy * s;
            const x2 = tx * s, y2 = ty * s;

            // Bright red beam line
            beam.clear();
            beam.alpha = alpha * 0.9;
            beam.lineStyle(Math.max(1, s * 2.5), 0xFF1133, 1);
            beam.moveTo(x1, y1);
            beam.lineTo(x2, y2);
            // Soft outer glow line
            beam.lineStyle(Math.max(2, s * 5), 0xFF4466, 0.3);
            beam.moveTo(x1, y1);
            beam.lineTo(x2, y2);

            // Impact ring at target
            flash.clear();
            flash.alpha = alpha * alpha;
            const r = s * 0.55 * (0.4 + 0.6 * (1 - t));
            flash.lineStyle(Math.max(1, s * 1.5), 0xFF4466, 1);
            flash.drawCircle(x2, y2, r);
            flash.lineStyle(0);
            flash.beginFill(0xFF1133, 0.35);
            flash.drawCircle(x2, y2, r * 0.55);
            flash.endFill();
        }, () => {
            container.removeChild(beam);
            container.removeChild(flash);
        });
    }
}

export class SpritesheetFrameAnimation extends FrameAnimation {
    constructor(options) {
        const start = options.start;
        const duration = options.duration || 2;
        const sheet = options.sheet;
        const x = options.x;
        const y = options.y;
        const sizeFactor = options.sizeFactor;
        const tintColor = options.tintColor;
        const cellSize = options.cellSize;
        const container = options.container;
        const opacity = options.opacity || 1.0;

        let frames = [];
        for (let frame of Object.keys(sheet.data.frames).sort()) {
            frames.push(PIXI.Texture.fromFrame(frame));
        }

        let sprite = PIXI.Sprite.from(frames[0]);
        sprite.anchor.x = 0.5;
        sprite.anchor.y = 0.5;
        sprite.position.x = cellSize * x + cellSize / 2;
        sprite.position.y = cellSize * y + cellSize / 2;
        sprite.width = cellSize * sizeFactor;
        sprite.height = cellSize * sizeFactor;
        sprite.tint = tintColor;
        sprite.alpha = opacity;
        sprite.blendMode = PIXI.BLEND_MODES.SCREEN;
        container.addChild(sprite);

        super(start, duration, () => {

        }, (camera, frameTime) => {
            const t = (duration - frameTime) / duration;
            const index = Math.floor(t * (frames.length - 1));
            sprite.width = cellSize * sizeFactor * camera.scale;
            sprite.height = cellSize * sizeFactor * camera.scale;
            const [ cellX, cellY ] = camera.worldToCamera(x, y);
            sprite.position.x = cellSize * cellX * camera.scale + cellSize * camera.scale / 2;
            sprite.position.y = cellSize * cellY * camera.scale + cellSize * camera.scale / 2;
            sprite.texture = frames[index];
            sprite.alpha = frameTime / duration;
        }, () => {
            container.removeChild(sprite);
        });
    }
}
