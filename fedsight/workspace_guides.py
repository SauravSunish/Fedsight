"""Plain-language help describing the features of each workspace."""

GUIDES = {
    "Overview": (
        "Start and organize an investigation.",
        "Shows the active case, evidence totals, review workload and shortcuts to the main tools.",
        ["Create a case", "Open an existing case", "Load the synthetic walkthrough", "Add case notes and manage case status"],
        "Start a separate case for footage from a particular location or incident.",
        "Create or select a case, then open Evidence & processing to add recordings.",
        "The walkthrough uses synthetic data. Tracklet totals count tracked appearances, not unique people."),
    "Evidence & processing": (
        "Turn original recordings into searchable sightings.",
        "Preserves uploaded originals, then runs the selected detector and tracker to extract tracklets and preview images.",
        ["Register videos and source cameras", "Document recording start times", "Configure detection and tracking", "Monitor jobs and retry failed sources"],
        "Upload two camera recordings before searching for appearances across them.",
        "Upload a supported video, preserve and register it, select processing settings, then click Process videos.",
        "Processing can miss people or split one appearance into several tracklets. Uploading a video runs inference; it does not train a model."),
    "Video review": (
        "Check a finding against the actual recording.",
        "Connects indexed tracklets and events to video frames, with optional detection boxes and other overlays.",
        ["Play source footage", "Jump to a tracklet or event", "Inspect an annotated frame", "Download snapshots and render annotated clips"],
        "Verify whether a search result really shows the clothing or object you are looking for.",
        "Use Camera wall to compare recordings at the same clip offset. Choose Single recording to jump to a sighting, check the original frame or export a clip.",
        "Boxes between nearby samples may be interpolated. Missing boxes do not prove that nobody was present."),
    "ThreadSight review": (
        "Decide whether two recorded appearances should be linked.",
        "Presents candidate pairs with images, source details, appearance scores and route context for human review.",
        ["Compare sightings side by side", "Read the score legend", "Filter pending decisions", "Accept, reject or defer a proposed link"],
        "Compare a person leaving one camera with a possible sighting in another recording.",
        "Inspect both source clips, read the score explanations, then record a decision and your reason.",
        "A high similarity score is not proof of identity. Defer a decision when evidence is insufficient."),
    "Search & digest": (
        "Find useful sightings and assemble a reviewable evidence summary.",
        "Filters indexed tracklets and provides source-linked previews so you can narrow a case without watching every recording again.",
        ["Filter sightings", "Compare appearance descriptors", "Build a preview review reel", "Prepare a case export"],
        "Find relevant sightings in a time interval and collect them for a presentation or review.",
        "Set filters, inspect the returned previews, and open source footage before exporting findings.",
        "A preview reel summarizes indexed images; it is not a continuous reconstruction of events."),
    "FedSight Investigator": (
        "Ask questions about the active case in plain language.",
        "Routes questions to case-search tools. Optional image models inspect previews or rank text-to-image candidates when configured.",
        ["Search recorded sightings and events", "Ask for a case summary", "Open source-linked results", "Start fresh chats and reopen previous chats"],
        "Ask 'Show restricted zone events' or search for a visible clothing description.",
        "Select the case, start a new conversation, and ask a specific question. Check each result's recording and timestamp.",
        "Visual search depends on installed models and indexing. Returned candidates and model observations require verification."),
    "Cases": (
        "Manage local cases and keep their findings and notes together.",
        "Collects bookmarked sightings and saved investigation material into a case board, with source references retained.",
        ["Create a case or edit its details", "Review bookmarks, notes and reports", "Delete an unwanted case and its evidence files", "Compact the database to reclaim unused space"],
        "Keep the strongest findings in one place before discussing them with a reviewer.",
        "Use Add to case or bookmark controls while reviewing sightings, then return here to inspect the collected material.",
        "Saving an item records it for review; it does not confirm the model's conclusion."),
    "Timeline & graph": (
        "Understand the order of sightings and the links between them.",
        "Displays time-based evidence and appearance groups formed from accepted human review decisions.",
        ["Explore the case timeline", "Inspect accepted appearance groups", "Run supported evidence queries", "Optionally project records to a configured Neo4j server"],
        "Follow a reviewed appearance group across several recordings and check gaps in its timeline.",
        "Inspect source times and accepted links. Align camera clocks before interpreting cross-camera order.",
        "Unaligned timestamps are relative to individual clips. An appearance group is not a verified real-world identity."),
    "Camera intelligence": (
        "Describe how cameras connect and how long travel between them might take.",
        "Stores camera information and directed transitions that support route and timing checks elsewhere in the app.",
        ["Register cameras", "Configure directed camera transitions", "Set travel-time bounds", "Inspect camera connectivity"],
        "Record that someone can travel from the entrance camera to the corridor camera through a connecting hallway.",
        "Register cameras, assign recordings to them, and add transitions with realistic minimum and maximum travel times.",
        "These routes are investigator configuration. A missing connection means unknown feasibility, not impossible movement."),
    "BlindSight": (
        "Investigate what could have happened between two visible sightings.",
        "Uses configured camera coverage, entry/exit portals, timing and route constraints to generate possible movement hypotheses across unseen intervals.",
        ["Align recording clocks", "Mark coverage and blind zones", "Configure camera-edge portals", "Review route hypotheses and investigator-recorded supporting or contradictory clues"],
        "A person disappears near a corridor entrance and a similar appearance later exits into another camera's view.",
        "Map the cameras and blind zones, align the recording times, mark entry/exit portals, then generate and review missing-interval hypotheses.",
        "A hypothesis is a possible explanation. It does not establish presence in an unseen area or prove the two sightings show the same person."),
    "Spatial intelligence": (
        "Relate image movement to a floor plan and assess travel feasibility.",
        "Uses configured planar homography calibration to project image positions onto a ground plane and inspect possible movement between cameras.",
        ["Configure four-point homography calibration", "Inspect projected floor-plan trajectories", "Compare cross-camera transit timing", "Review physical feasibility estimates"],
        "Check whether the time between two appearances is compatible with the estimated ground-plane distance.",
        "Provide corresponding image and floor-plan points, check the calibration, then inspect projected paths and transit results.",
        "Projection depends on accurate calibration and a suitable ground-plane assumption. Physical feasibility does not establish identity."),
    "Rules & events": (
        "Flag recorded activity that meets a condition you define.",
        "Applies configured rules to sampled detections and trajectories, then saves candidate events for review.",
        ["Configure zones and line crossings", "Define dwell, crowd or motion conditions", "Enable or disable rules", "Review generated event flags"],
        "Flag a tracked appearance entering a restricted zone in recorded footage.",
        "Create a rule for the relevant camera, run the event evaluation, then check flagged moments in Video review.",
        "Flags depend on detections and sampling. They do not infer intent; apparent unattended objects can have missed nearby people."),
    "Live respond": (
        "Try live detection and compatible rules using a local camera.",
        "Samples camera images at the dashboard refresh rate and displays detection or alert feedback in the current browser session.",
        ["Use a local camera", "Inspect sampled live detections", "Apply compatible configured rules", "View session alert history"],
        "Demonstrate a configured zone or counting rule with a local camera.",
        "Configure compatible rules in Rules & events, select the local camera options here, and start the live demonstration.",
        "This is a local dashboard demonstration, not an always-on monitoring service. Alert history is session-based; external notifications are not provided."),
    "Analytics": (
        "Explore patterns in the activity the app has indexed.",
        "Aggregates sampled tracklets into charts of activity, stationary visibility and movement within each camera image.",
        ["View activity trends", "Compare recording activity", "Inspect stationary visibility", "Explore motion density and common image-plane paths"],
        "Find which portions of a recording contain more activity or where tracked appearances remain stationary longest.",
        "Choose the recordings and available filters, then inspect the charts and verify interesting periods in Video review.",
        "Track breaks can inflate counts. Motion density is not confirmed occupancy, and image-plane paths from different cameras are not automatically merged."),
    "Evaluation lab": (
        "Measure a matching experiment using labelled examples.",
        "Compares pairwise classifiers with separate training, calibration, validation and test splits, and reports errors and calibration metrics.",
        ["Upload labelled pair features", "Compare classifier experiments", "Inspect held-out metrics and errors", "Run a synthetic software check or federated research sandbox"],
        "Compare matching decisions on independently labelled examples before relying on a threshold.",
        "Use the dataset template, prepare independent splits, then train and evaluate. Read the test errors as well as the headline scores.",
        "This trains pairwise classifiers, not the video detector or OSNet backbone. Synthetic scores test software behavior and do not measure real-world accuracy."),
    "Integrity & audit": (
        "Check stored evidence integrity and inspect the record of actions.",
        "Verifies authenticated case storage and audit history, with exportable signed checkpoints for independent retention.",
        ["Check integrity", "Inspect audit events", "Export or verify a signed checkpoint", "Review local replay-detection tools"],
        "Verify that stored evidence can be authenticated before preparing a case export.",
        "Run the integrity checks, inspect any reported issue, and retain an exported checkpoint separately from the case store.",
        "Integrity checks do not establish that a recording depicts a true event. Local reviewer names are attribution labels, not authenticated accounts."),
}


def show_workspace_guide(st, page):
    purpose, operation, features, example, first_step, limitation = GUIDES[page]
    with st.expander(f"About {page} · purpose, features & how to use", expanded=False):
        st.write(purpose)
        st.markdown("**How it works**")
        st.write(operation)
        st.markdown("**What you can do**")
        st.markdown("\n".join(f"- {feature}" for feature in features))
        st.markdown("**Example use case**")
        st.write(example)
        st.markdown("**Get started**")
        st.write(first_step)
        st.caption(limitation)
