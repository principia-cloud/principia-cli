# IMPORTANT: `npm run package:brew` to update this file after publishing a new version of the package
class Principia < Formula
  desc "AI agent for robotics simulation"
  homepage "https://principia.cloud"
  url "https://registry.npmjs.org/principia/-/principia-0.0.1.tgz" # TODO: update after first npm publish
  sha256 "0000000000000000000000000000000000000000000000000000000000000000" # TODO: update after first npm publish
  license "Apache-2.0"

  depends_on "node@20"
  depends_on "ripgrep"

  def install
    system "npm", "install", *std_npm_args(prefix: false)
    bin.install_symlink Dir["#{libexec}/bin/*"]
  end

  test do
    # Test that the binary exists and is executable
    assert_match version.to_s, shell_output("#{bin}/principia --version")
  end
end
