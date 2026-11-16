import java.io.FileWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;

import org.matsim.api.core.v01.population.Person;
import org.matsim.api.core.v01.population.Plan;
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
import org.matsim.core.controler.events.IterationEndsEvent;
import org.matsim.core.controler.events.StartupEvent;
import org.matsim.core.controler.listener.AfterMobsimListener;
import org.matsim.core.controler.listener.BeforeMobsimListener;
import org.matsim.core.controler.listener.IterationEndsListener;
import org.matsim.core.controler.listener.StartupListener;
import org.matsim.core.scenario.ScenarioUtils;

/**
 * MATSim with replanning for a fixed wall-clock time, for the weaving showcase.
 *
 * <pre>
 * java -cp showcase/java/classes:$(cat verification/java/target/classpath.txt) ReplanRun \
 *     --config config.xml --output out/ --record record.csv --time-limit 300 \
 *     --strategy ChangeExpBeta=0.7 --strategy ReRoute=0.15 [--set module.param=value ...]
 * </pre>
 *
 * Runs the config's iterations with the given strategies (in place of the
 * config's) and no output files. After each iteration it appends to
 * {@code --record}: the wall time since startup, the mobsim's start and end
 * (seconds since startup), and the mean score of the executed (selected)
 * plans and of each person's best plan. It exits after the first iteration
 * that ends past {@code --time-limit} seconds.
 */
public final class ReplanRun {

	public static void main(String[] args) throws IOException {
		String configPath = null, output = null, record = null;
		double timeLimit = Double.POSITIVE_INFINITY;
		List<String> overrides = new ArrayList<>(), strategies = new ArrayList<>();
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
				case "--config" -> configPath = args[++i];
				case "--output" -> output = args[++i];
				case "--record" -> record = args[++i];
				case "--time-limit" -> timeLimit = Double.parseDouble(args[++i]);
				case "--set" -> overrides.add(args[++i]);
				case "--strategy" -> strategies.add(args[++i]);
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
		if (!strategies.isEmpty()) {
			config.replanning().clearStrategySettings();
			for (String s : strategies) {
				int eq = s.indexOf('=');
				StrategySettings st = new StrategySettings();
				st.setStrategyName(s.substring(0, eq));
				st.setWeight(Double.parseDouble(s.substring(eq + 1)));
				config.replanning().addStrategySettings(st);
			}
		}

		final double limit = timeLimit;
		final long[] t = new long[3]; // startup, mobsim start, mobsim end
		final FileWriter w = new FileWriter(record);
		w.write("iteration,wall_s,mobsim_start_s,mobsim_end_s,score_executed,score_best\n");
		w.flush();
		Controler controler = new Controler(ScenarioUtils.loadScenario(config));
		controler.addOverridingModule(new AbstractModule() {
			@Override
			public void install() {
				addControlerListenerBinding().toInstance((StartupListener) e -> t[0] = System.nanoTime());
				addControlerListenerBinding().toInstance(new BeforeMobsimListener() {
					@Override public void notifyBeforeMobsim(BeforeMobsimEvent e) { t[1] = System.nanoTime(); }
					@Override public double priority() { return -1e9; } // last
				});
				addControlerListenerBinding().toInstance(new AfterMobsimListener() {
					@Override public void notifyAfterMobsim(AfterMobsimEvent e) { t[2] = System.nanoTime(); }
					@Override public double priority() { return 1e9; } // first
				});
				addControlerListenerBinding().toInstance(new IterationEndsListener() {
					@Override public void notifyIterationEnds(IterationEndsEvent e) {
						double executed = 0, best = 0;
						int n = 0;
						for (Person p : e.getServices().getScenario().getPopulation().getPersons().values()) {
							Double s = p.getSelectedPlan().getScore();
							double b = Double.NEGATIVE_INFINITY;
							for (Plan plan : p.getPlans()) {
								if (plan.getScore() != null) b = Math.max(b, plan.getScore());
							}
							executed += s == null ? 0 : s;
							best += b;
							n++;
						}
						double wall = (System.nanoTime() - t[0]) / 1e9;
						try {
							w.write(e.getIteration() + "," + wall + "," + (t[1] - t[0]) / 1e9 + ","
									+ (t[2] - t[0]) / 1e9 + "," + executed / n + "," + best / n + "\n");
							w.flush();
							if (wall >= limit) {
								w.close();
								System.exit(0); // the time is up: stop without MATSim's shutdown output
							}
						} catch (IOException ex) {
							throw new UncheckedIOException(ex);
						}
					}
					@Override public double priority() { return -1e9; } // after MATSim's own
				});
			}
		});
		controler.run();
		w.close();
	}
}
