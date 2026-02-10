import { Box, Text } from "ink"
// biome-ignore lint/style/useImportType: React is used as a value by the classic JSX transform (jsxFactory: React.createElement)
import React from "react"

const PRINCIPIA_LOGO = [
	"                               ██████                              ",
	"                         ██████████████████                        ",
	"                      ████████████████████████                     ",
	"                    ████████████████████████████                   ",
	"                   ██████████████████████████████                  ",
	"                   ██████████████████████████████                  ",
	"                   ██████████████████████████████                  ",
	"                   ██████████████████████████████                  ",
	"          █████████                              █████████         ",
	"         ██████████                              ██████████        ",
	"        ███████████                              ███████████       ",
	"        ███████████                              ███████████       ",
	"       ████████████       ███          ███       ████████████      ",
	"       ████████████     ██████        ██████     ████████████      ",
	"      █████████████     ██████        ██████     █████████████     ",
	"      █████████████     ██████        ██████     █████████████     ",
	"      █████████████     ██████        ██████     █████████████     ",
	"      █████████████                              █████████████     ",
	"      █████████████                              █████████████     ",
	"      █████████████                              █████████████     ",
	"      █████████████                              ████████████      ",
	"       ████████████                              ███████████       ",
	"        ███████████                              ███████████       ",
	"         ██████████                              █████████         ",
	"          █████████                              █████████         ",
	"                   ██████████████████████████████                  ",
	"                   ██████████████████████████████                  ",
	"                   ██████████████████████████████                  ",
	"                   ██████████████████████████████                  ",
	"                    ███████████████████████████                    ",
	"                       ██████████████████████                      ",
	"                           ██████████████                          ",
	"                                                                   ",
	"   ██████  ██████  ██  ██    ██   ██████ ██  ██████  ██   █████   ",
	"   ██  ██  ██  ██  ██  ███   ██  ██      ██  ██  ██  ██  ██   ██  ",
	"   ██████  ██████  ██  ██ ██ ██  ██      ██  ██████  ██  ███████  ",
	"   ██      ██  ██  ██  ██  ███   ██      ██  ██      ██  ██   ██  ",
	"   ██      ██  ██  ██  ██    ██   ██████ ██  ██      ██  ██   ██  ",
]

// Gradient colors for each line (cyan -> blue -> purple -> magenta -> red)
const LOGO_GRADIENT_COLORS = [
	"#00D9FF",
	"#00D2FF",
	"#00CBFF",
	"#00C2FF",
	"#00B8FF",
	"#00AEFF",
	"#1DA5FF",
	"#3A9BFF",
	"#4D9FFF",
	"#5A96FF",
	"#6B8EFF",
	"#7B8FFF",
	"#8A86FF",
	"#9B7FFF",
	"#AE72FF",
	"#B96FFF",
	"#C268FF",
	"#CB62FF",
	"#D45CFF",
	"#DD58FF",
	"#E94FFF",
	"#EE4AFF",
	"#F540FF",
	"#F83DFF",
	"#FC38FF",
	"#FF42F0",
	"#FF45E2",
	"#FF4BC8",
	"#FF4EBB",
	"#FF53A5",
	"#FF5892",
	"#FF6078",
	"#FF6668",
	"#00D9FF",
	"#00C4FF",
	"#00AFFF",
	"#4D9FFF",
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
