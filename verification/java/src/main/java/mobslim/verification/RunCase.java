package mobslim.verification;

import java.util.ArrayList;
import java.util.List;

import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigGroup;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.controler.Controler;
import org.matsim.core.controler.OutputDirectoryHierarchy;
import org.matsim.core.scenario.ScenarioUtils;

/**
 * Runs one MATSim verification case.
 *
 * <pre>
 * java -cp target/classes:$(cat target/classpath.txt) mobslim.verification.RunCase \
 *     --config config.xml --output out/ [--set module.param=value ...]
 * </pre>
 *
 * The Python side writes a complete config per case (scoring variant, paths);
 * {@code --set} only overrides plain top-level module params (iterations,
 * mobsim, seed, capacity factors).
 */
public final class RunCase {

	public static void main(String[] args) {
		String configPath = null;
		String output = null;
		List<String> overrides = new ArrayList<>();
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
				case "--config" -> configPath = args[++i];
				case "--output" -> output = args[++i];
				case "--set" -> overrides.add(args[++i]);
				default -> throw new IllegalArgumentException("Unknown argument: " + args[i]);
			}
		}
		if (configPath == null || output == null) {
			throw new IllegalArgumentException("--config and --output are required");
		}

		Config config = ConfigUtils.loadConfig(configPath);
		for (String o : overrides) {
			int eq = o.indexOf('=');
			int dot = o.indexOf('.');
			if (eq < 0 || dot < 0 || dot > eq) {
				throw new IllegalArgumentException("--set expects module.param=value, got: " + o);
			}
			ConfigGroup module = config.getModules().get(o.substring(0, dot));
			if (module == null) {
				throw new IllegalArgumentException("Unknown config module in: " + o);
			}
			module.addParam(o.substring(dot + 1, eq), o.substring(eq + 1));
		}
		config.controller().setOutputDirectory(output);
		config.controller().setOverwriteFileSetting(
				OutputDirectoryHierarchy.OverwriteFileSetting.deleteDirectoryIfExists);

		Controler controler = new Controler(ScenarioUtils.loadScenario(config));
		controler.run();
	}
}
