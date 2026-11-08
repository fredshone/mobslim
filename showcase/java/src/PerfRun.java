import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigGroup;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.config.groups.ControllerConfigGroup;
import org.matsim.core.config.groups.ReplanningConfigGroup.StrategySettings;
import org.matsim.core.controler.AbstractModule;
import org.matsim.core.controler.Controler;
import org.matsim.core.controler.OutputDirectoryHierarchy;
import org.matsim.core.controler.events.AfterMobsimEvent;
import org.matsim.core.controler.events.BeforeMobsimEvent;
import org.matsim.core.controler.listener.AfterMobsimListener;
import org.matsim.core.controler.listener.BeforeMobsimListener;
import org.matsim.core.scenario.ScenarioUtils;

/**
 * Times MATSim's mobsim for the showcase benchmarks.
 *
 * <pre>
 * java -cp showcase/java/classes:$(cat verification/java/target/classpath.txt) PerfRun \
 *     --config config.xml --output out/ --timings timings.csv [--set module.param=value ...]
 * </pre>
 *
 * Runs the plans as given (routed) for {@code lastIteration + 1} iterations
 * with no replanning (KeepLastSelected) and no output files, and writes the
 * wall time of each iteration's mobsim (between the last BeforeMobsim and the
 * first AfterMobsim listener, nanosecond clock) to {@code --timings}.
 */
public final class PerfRun {

	public static void main(String[] args) throws IOException {
		String configPath = null, output = null, timings = null;
		List<String> overrides = new ArrayList<>();
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
				case "--config" -> configPath = args[++i];
				case "--output" -> output = args[++i];
				case "--timings" -> timings = args[++i];
				case "--set" -> overrides.add(args[++i]);
				default -> throw new IllegalArgumentException("Unknown argument: " + args[i]);
			}
		}
		Config config = ConfigUtils.loadConfig(configPath);
		for (String o : overrides) {
			int eq = o.indexOf('='), dot = o.indexOf('.');
			ConfigGroup module = config.getModules().get(o.substring(0, dot));
			if (module == null) throw new IllegalArgumentException("Unknown config module in: " + o);
			module.addParam(o.substring(dot + 1, eq), o.substring(eq + 1));
		}
		config.controller().setOutputDirectory(output);
		config.controller().setOverwriteFileSetting(
				OutputDirectoryHierarchy.OverwriteFileSetting.deleteDirectoryIfExists);
		config.controller().setWriteEventsInterval(0);
		config.controller().setWritePlansInterval(0);
		config.controller().setWriteTripsInterval(0);
		config.controller().setCreateGraphsInterval(0);
		config.controller().setDumpDataAtEnd(false);
		config.controller().setCleanItersAtEnd(ControllerConfigGroup.CleanIterations.delete);
		config.replanning().clearStrategySettings();
		StrategySettings keep = new StrategySettings();
		keep.setStrategyName("KeepLastSelected");
		keep.setWeight(1.0);
		config.replanning().addStrategySettings(keep);

		final List<long[]> rows = new ArrayList<>();
		final long[] start = new long[1];
		Controler controler = new Controler(ScenarioUtils.loadScenario(config));
		controler.addOverridingModule(new AbstractModule() {
			@Override
			public void install() {
				addControlerListenerBinding().toInstance(new BeforeMobsimListener() {
					@Override public void notifyBeforeMobsim(BeforeMobsimEvent e) { start[0] = System.nanoTime(); }
					@Override public double priority() { return -1e9; } // last
				});
				addControlerListenerBinding().toInstance(new AfterMobsimListener() {
					@Override public void notifyAfterMobsim(AfterMobsimEvent e) {
						rows.add(new long[] {e.getIteration(), System.nanoTime() - start[0]});
					}
					@Override public double priority() { return 1e9; } // first
				});
			}
		});
		controler.run();
		try (FileWriter w = new FileWriter(timings)) {
			w.write("iteration,mobsim_ns\n");
			for (long[] r : rows) w.write(r[0] + "," + r[1] + "\n");
		}
	}
}
