"use client";

import { useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface AnimatedGradientBackgroundProps {
    className?: string;
    children?: React.ReactNode;
    intensity?: "subtle" | "medium" | "strong";
}

interface Beam {
    x: number;
    y: number;
    width: number;
    length: number;
    angle: number;
    speed: number;
    opacity: number;
    hue: number;
    pulse: number;
    pulseSpeed: number;
}

function createBeam(width: number, height: number): Beam {
    const angle = -35 + Math.random() * 10;
    return {
        x: Math.random() * width * 1.5 - width * 0.25,
        y: Math.random() * height * 1.5 - height * 0.25,
        width: 40 + Math.random() * 80,
        length: height * 2.5,
        angle: angle,
        speed: 0.4 + Math.random() * 0.8,
        opacity: 0.2 + Math.random() * 0.2,
        hue: 210 + Math.random() * 20, // Professional soft blue
        pulse: Math.random() * Math.PI * 2,
        pulseSpeed: 0.005 + Math.random() * 0.01, // Slower pulse
    };
}

export function BeamsBackground({
    className,
    children,
    intensity = "subtle",
}: AnimatedGradientBackgroundProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const beamsRef = useRef<Beam[]>([]);
    const animationFrameRef = useRef<number>(0);
    const MINIMUM_BEAMS = 20;

    const opacityMap = {
        subtle: 0.7,
        medium: 0.85,
        strong: 1,
    };

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const updateCanvasSize = () => {
            const dpr = window.devicePixelRatio || 1;
            canvas.width = window.innerWidth * dpr;
            canvas.height = window.innerHeight * dpr;
            canvas.style.width = `${window.innerWidth}px`;
            canvas.style.height = `${window.innerHeight}px`;
            ctx.scale(dpr, dpr);

            const totalBeams = MINIMUM_BEAMS * 1.5;
            beamsRef.current = Array.from({ length: totalBeams }, () =>
                createBeam(canvas.width, canvas.height)
            );
        };

        updateCanvasSize();
        window.addEventListener("resize", updateCanvasSize);

        function resetBeam(beam: Beam, index: number, totalBeams: number) {
            if (!canvas) return beam;

            const column = index % 3;
            const spacing = canvas.width / 3;

            beam.y = canvas.height + 100;
            beam.x =
                column * spacing +
                spacing / 2 +
                (Math.random() - 0.5) * spacing * 0.5;
            beam.width = 120 + Math.random() * 80;
            beam.speed = 0.4 + Math.random() * 0.3;
            beam.hue = 240 + (index * 40) / totalBeams;
            beam.opacity = 0.25 + Math.random() * 0.15;
            return beam;
        }

        function drawBeam(ctx: CanvasRenderingContext2D, beam: Beam) {
            ctx.save();
            ctx.translate(beam.x, beam.y);
            ctx.rotate((beam.angle * Math.PI) / 180);

            // Calculate pulsing opacity
            const pulsingOpacity =
                beam.opacity *
                (0.8 + Math.sin(beam.pulse) * 0.2) *
                opacityMap[intensity];

            const gradient = ctx.createLinearGradient(0, 0, 0, beam.length);

            // Enhanced gradient with multiple color stops
            gradient.addColorStop(0, `hsla(${beam.hue}, 85%, 60%, 0)`);
            gradient.addColorStop(
                0.1,
                `hsla(${beam.hue}, 85%, 60%, ${pulsingOpacity * (intensity === "subtle" ? 0.05 : 0.1)})`
            );
            gradient.addColorStop(
                0.4,
                `hsla(${beam.hue}, 85%, 60%, ${pulsingOpacity * 0.2})`
            );
            gradient.addColorStop(
                0.6,
                `hsla(${beam.hue}, 85%, 60%, ${pulsingOpacity * 0.2})`
            );
            gradient.addColorStop(
                0.9,
                `hsla(${beam.hue}, 85%, 60%, ${pulsingOpacity * (intensity === "subtle" ? 0.05 : 0.1)})`
            );
            gradient.addColorStop(1, `hsla(${beam.hue}, 85%, 60%, 0)`);

            ctx.shadowBlur = 40;
            ctx.shadowColor = `hsla(${beam.hue}, 80%, 60%, ${beam.opacity * 0.5})`;
            ctx.fillStyle = gradient;
            ctx.fillRect(-beam.width / 2, 0, beam.width, beam.length);
            ctx.restore();
        }

        function animate() {
            if (!canvas || !ctx) return;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.filter = "blur(35px)";

            const totalBeams = beamsRef.current.length;
            beamsRef.current.forEach((beam, index) => {
                beam.y -= beam.speed;
                beam.pulse += beam.pulseSpeed;

                // Reset beam when it goes off screen
                if (beam.y + beam.length < -100) {
                    resetBeam(beam, index, totalBeams);
                }

                drawBeam(ctx, beam);
            });

            animationFrameRef.current = requestAnimationFrame(animate);
        }

        animate();

        return () => {
            window.removeEventListener("resize", updateCanvasSize);
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
        };
    }, [intensity]);

    return (
        <div
            className={cn(
                "relative min-h-screen w-full overflow-hidden bg-neutral-50",
                className
            )}
        >
            <canvas
                ref={canvasRef}
                className="absolute inset-0 pointer-events-none opacity-30 mix-blend-multiply"
                style={{ filter: "blur(8px)" }}
            />
            <div className="absolute inset-0 bg-gradient-to-tr from-blue-50/20 via-transparent to-orange-50/10 pointer-events-none" />

            {children}
            <div
                className="absolute inset-0 bg-white/20 pointer-events-none"
                style={{
                    boxShadow: "inset 0 0 100px rgba(255,255,255,0.5)"
                }}
            />

        </div>
    );
}
