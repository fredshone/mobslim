package mobslim.verification;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;

import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.population.Person;
import org.matsim.core.api.experimental.events.EventsManager;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.events.EventsUtils;
import org.matsim.core.events.MatsimEventsReader;
import org.matsim.core.scenario.ScenarioUtils;
import org.matsim.core.scoring.EventsToScore;
import org.matsim.core.scoring.functions.CharyparNagelScoringFunctionFactory;

/**
 * Scores an events file with MATSim's default (Charypar-Nagel) scoring.
 *
 * <pre>
 * java -cp ... mobslim.verification.ScoreEvents \
 *     --config config.xml --plans plans.xml.gz --events events.xml.gz --output scores.csv
 * </pre>
 *
 * The config supplies the network, transit and scoring parameters; the plans
 * file the persons (and their activity types). Writes {@code person,score}.
 */
public final class ScoreEvents {

	public static void main(String[] args) throws IOException {
		String configPath = null, plans = null, events = null, output = null;
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
				case "--config" -> configPath = args[++i];
				case "--plans" -> plans = args[++i];
				case "--events" -> events = args[++i];
				case "--output" -> output = args[++i];
				default -> throw new IllegalArgumentException("Unknown argument: " + args[i]);
			}
		}
		if (configPath == null || plans == null || events == null || output == null) {
			throw new IllegalArgumentException("--config, --plans, --events and --output are required");
		}
		Config config = ConfigUtils.loadConfig(configPath);
		config.plans().setInputFile(Path.of(plans).toAbsolutePath().toString());
		Scenario scenario = ScenarioUtils.loadScenario(config);

		EventsManager manager = EventsUtils.createEventsManager();
		EventsToScore scoring = EventsToScore.createWithoutScoreUpdating(
				scenario, new CharyparNagelScoringFunctionFactory(scenario), manager);
		scoring.beginIteration(0, true);
		manager.initProcessing();
		new MatsimEventsReader(manager).readFile(events);
		manager.finishProcessing();
		scoring.finish();

		Path out = Path.of(output);
		Files.createDirectories(out.toAbsolutePath().getParent());
		try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(out))) {
			w.println("person,score");
			for (Person person : scenario.getPopulation().getPersons().values()) {
				Double score = scoring.getAgentScore(person.getId());
				w.println(person.getId() + "," + (score == null ? "" : score));
			}
		}
	}
}
