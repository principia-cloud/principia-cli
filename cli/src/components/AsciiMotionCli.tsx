import { Box, Text } from "ink"
// biome-ignore lint/style/useImportType: React is used as a value by the classic JSX transform (jsxFactory: React.createElement)
import React from "react"

const PRINCIPIA_LOGO = [
	"                    ███                    ",
	"               █████████████               ",
	"             █████████████████             ",
	"            ███████████████████            ",
	"            ███████████████████            ",
	"         █████               █████         ",
	"        ██████               ██████        ",
	"        ██████   ███   ███   ██████        ",
	"        ██████   ███   ███   ██████        ",
	"        ██████               ██████        ",
	"         █████               █████         ",
	"            ███████████████████            ",
	"            ███████████████████            ",
	"             █████████████████             ",
	"                ███████████                ",
	"                                           ",
	"████  ████  █  █  █   ███  █  ████  █   ██ ",
	"█  █  █  █  █  ██ █  █     █  █  █  █  █  █",
	"████  ███   █  █ ██  █     █  ████  █  ████",
	"█     █  █  █  █  █   ███  █  █     █  █  █",
]

// Gradient colors for each line (cyan -> blue -> purple -> magenta -> red)
const LOGO_GRADIENT_COLORS = [
	"#00D9FF",
	"#00CBFF",
	"#00B8FF",
	"#1DA5FF",
	"#3A9BFF",
	"#5A96FF",
	"#7B8FFF",
	"#9B7FFF",
	"#AE72FF",
	"#C268FF",
	"#D45CFF",
	"#EE4AFF",
	"#F83DFF",
	"#FF42F0",
	"#FF4BC8",
	"#FF5892",
	"#00D9FF",
	"#00AFFF",
	"#9B7FFF",
	"#FF3FFF",
]

type AsciiMotionCliProps = {
	hasDarkBackground?: boolean
	autoPlay?: boolean
	loop?: boolean
	onReady?: (api: { play: () => void; pause: () => void; restart: () => void }) => void
	onInteraction?: () => void
}

/**
 * Static single-frame robot logo with gradient colors
 */
export const StaticRobotFrame: React.FC = () => {
	return (
		<Box alignItems="center" flexDirection="column">
			{PRINCIPIA_LOGO.map((line, idx) => (
				<Text color={LOGO_GRADIENT_COLORS[idx] || "#FF7540"} key={idx}>
					{line}
				</Text>
			))}
		</Box>
	)
}

/**
 * Principia Logo Component - Robot with gradient colors
 */
export const AsciiMotionCli: React.FC<AsciiMotionCliProps> = ({ onInteraction }) => {
	return (
		<Box alignItems="center" flexDirection="column" width="100%">
			<Box alignItems="center" flexDirection="column">
				{PRINCIPIA_LOGO.map((line, idx) => (
					<Text color={LOGO_GRADIENT_COLORS[idx] || "#FF7540"} key={idx}>
						{line}
					</Text>
				))}
				<Text> </Text>
				<Text bold color="#00D9FF">
					AI Agent for Robotics Simulation
				</Text>
			</Box>
		</Box>
	)
}
