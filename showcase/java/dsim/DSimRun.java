import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.matsim.api.core.v01.Scenario;
import org.matsim.core.communication.NullCommunicator;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigGroup;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.config.consistency.UnmaterializedConfigGroupChecker;
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
import org.matsim.core.serialization.NoopSerializationProvider;
import org.matsim.dsim.DistributedContext;

/**
 * Runs MATSim's DSim (weekly builds only) in one JVM, the way MATSim's own
 * {@code RunDistributedSim} does: a single-node {@link DistributedContext}.
 *
 * <pre>
 * java -cp ... DSimRun --config config.xml --output out/ [--timings t.csv] [--set module.param=value ...]
 * </pre>
 *
 * The {@code dsim} module is filled from the {@code qsim} one (FIFO links,
 * queue dynamics, its stuck time, start and end time), as in MATSim's
 * DSimIntegrationTest. Without {@code --timings} it runs the config as
 * written and writes events (like {@code RunCase}); with it, it runs
 * {@code lastIteration + 1} iterations with no replanning and no output and
 * writes each iteration's mobsim wall time (like {@code PerfRun}).
 */
public final class DSimRun {

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
		config.controller().setMobsim(ControllerConfigGroup.MobsimType.dsim.name());
		config.dsim().setLinkDynamics(config.qsim().getLinkDynamics());
		config.dsim().setTrafficDynamics(config.qsim().getTrafficDynamics());
		config.dsim().setStuckTime(config.qsim().getStuckTime());
		config.dsim().setStartTime(config.qsim().getStartTime().orElse(0));
		config.dsim().setEndTime(config.qsim().getEndTime().orElse(30 * 3600));
		config.dsim().setVehicleBehavior(config.qsim().getVehicleBehavior());
		config.routing().setRoutingRandomness(0);
		for (String o : overrides) {
			int eq = o.indexOf('='), dot = o.indexOf('.');
			ConfigGroup module = config.getModules().get(o.substring(0, dot));
			if (module == null) throw new IllegalArgumentException("Unknown config module in: " + o);
			module.addParam(o.substring(dot + 1, eq), o.substring(eq + 1));
		}
		config.controller().setOutputDirectory(output);
		config.controller().setOverwriteFileSetting(
				OutputDirectoryHierarchy.OverwriteFileSetting.deleteDirectoryIfExists);
		if (timings != null) {
			config.controller().setWriteEventsInterval(0);
			config.controller().setWritePlansInterval(0);
			config.controller().setWriteTripsInterval(0);
			config.controller().setCreateGraphsInterval(0);
			config.controller().setDumpDataAtEnd(false);
			config.replanning().clearStrategySettings();
			StrategySettings keep = new StrategySettings();
			keep.setStrategyName("KeepLastSelected");
			keep.setWeight(1.0);
			config.replanning().addStrategySettings(keep);
		}

		Scenario scenario = ScenarioUtils.loadScenario(config);
		Controler controler = new Controler(scenario,
				DistributedContext.create(new NullCommunicator(), config, new NoopSerializationProvider()));
		final List<long[]> rows = new ArrayList<>();
		final long[] start = new long[1];
		if (timings != null) {
			controler.addOverridingModule(new AbstractModule() {
				@Override
				public void install() {
					addControlerListenerBinding().toInstance(new BeforeMobsimListener() {
						@Override public void notifyBeforeMobsim(BeforeMobsimEvent e) { start[0] = System.nanoTime(); }
						@Override public double priority() { return -1e9; }
					});
					addControlerListenerBinding().toInstance(new AfterMobsimListener() {
						@Override public void notifyAfterMobsim(AfterMobsimEvent e) {
							rows.add(new long[] {e.getIteration(), System.nanoTime() - start[0]});
						}
						@Override public double priority() { return 1e9; }
					});
				}
			});
		}
		controler.getInjector();
		// as RunDistributedSim: the dsim groups are only materialised once the injector exists
		controler.getConfig().removeConfigConsistencyChecker(UnmaterializedConfigGroupChecker.class);
		controler.run();
		if (timings != null) {
			try (FileWriter w = new FileWriter(timings)) {
				w.write("iteration,mobsim_ns\n");
				for (long[] r : rows) w.write(r[0] + "," + r[1] + "\n");
			}
		}
	}
}
