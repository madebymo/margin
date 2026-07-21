import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

async function expectNoAccessibilityViolations(page, state) {
  const results = await new AxeBuilder({ page })
    .withTags([
      "wcag2a",
      "wcag2aa",
      "wcag21a",
      "wcag21aa",
      "wcag22aa",
    ])
    .analyze();
  const violations = results.violations.map((violation) => ({
    id: violation.id,
    impact: violation.impact,
    help: violation.help,
    targets: violation.nodes.map((node) => node.target),
  }));
  expect(violations, `${state} accessibility violations`).toEqual([]);
}

function tabKeyFor(page, { reverse = false } = {}) {
  const browserName = page.context().browser()?.browserType().name();
  const modifiers = [];
  if (process.platform === "darwin" && browserName === "webkit") {
    modifiers.push("Alt");
  }
  if (reverse) modifiers.push("Shift");
  return [...modifiers, "Tab"].join("+");
}

async function tabTo(page, target, limit = 40) {
  await expect(target).toBeVisible();
  const tabKey = tabKeyFor(page);
  for (let index = 0; index < limit; index += 1) {
    await page.keyboard.press(tabKey);
    if (await target.evaluate((node) => node === document.activeElement)) {
      return;
    }
  }
  throw new Error(`Keyboard focus did not reach ${await target.textContent()}`);
}

async function expectNoHorizontalOverflow(page) {
  const dimensions = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }));
  expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth + 1);
}

test("unified v2 lesson retries and completes without a page refresh", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", {
      name: "What would you like to work toward?",
    }),
  ).toBeVisible();
  await expect(page.getByLabel("Goal")).toHaveValue("goal.der.power_rule");
  await expectNoAccessibilityViolations(page, "desktop intake");

  const start = page.getByRole("button", {
    name: "Start my learning path",
  });
  await tabTo(page, start);
  await page.keyboard.press("Enter");

  const answer = page.getByLabel("Your expression");
  await expect(page.getByRole("heading", { name: "Power rule" })).toBeVisible();
  await expect(page.getByText("Engineering demo", { exact: true })).toBeVisible();
  await expect(page.getByText("Update 0")).toBeVisible();
  await expect(answer).toBeFocused();

  await page.keyboard.press(tabKeyFor(page));
  const hint = page.getByRole("button", { name: "Get a hint" });
  await expect(hint).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.getByText("Update 1")).toBeVisible();
  await expect(page.getByText("Hint", { exact: true })).toBeVisible();

  await page.keyboard.press(tabKeyFor(page, { reverse: true }));
  await expect(answer).toBeFocused();
  await page.keyboard.type("0");
  await page.keyboard.press("Enter");
  await expect(page.getByText("Update 2")).toBeVisible();
  await expect(answer).toBeFocused();

  await page.keyboard.type("0");
  await page.keyboard.press("Enter");
  await expect(page.getByText("Update 3")).toBeVisible();

  await expect(
    page.getByRole("button", { name: /text alternative/i }),
  ).toHaveCount(0);
  await expect(answer).toBeFocused();
  await page.keyboard.type("12*x^3");
  await page.keyboard.press("Enter");
  await expect(page.getByText("Update 4")).toBeVisible();
  await expect(
    page.getByText("Guided text practice complete. Now try an unseen check."),
  ).toBeVisible();
  await expect(page.getByLabel("Your expression")).toBeFocused();

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole("heading", { name: "Progress" })).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "What we know so far" }),
  ).toBeVisible();
  await expectNoHorizontalOverflow(page);
  await expectNoAccessibilityViolations(page, "mobile lesson");

  await page.setViewportSize({ width: 1280, height: 900 });
  await expect(page.locator(".assessment-bubble").last()).toContainText("2*x^5");
  const revisionBeforeRetry = await page.locator(".revision").textContent();
  let droppedCommittedResponse = false;
  await page.route("**/api/v2/sessions/*/actions", async (route) => {
    if (droppedCommittedResponse) {
      await route.continue();
      return;
    }
    droppedCommittedResponse = true;
    await route.fetch();
    await route.abort("failed");
  });

  await answer.fill("10*x^4");
  await answer.press("Enter");
  await expect(page.getByRole("alert")).toContainText(/could not be reached/i);
  await expect(page.locator(".revision")).toHaveText(revisionBeforeRetry);

  // The draft stays in the field and MutationCoordinator reuses the original
  // request id. The server replays the committed response without advancing a
  // second time.
  await answer.press("Enter");
  await expect(page.getByText("Update 5")).toBeVisible();
  await page.unroute("**/api/v2/sessions/*/actions");
  await expect(page.locator(".assessment-bubble").last()).toContainText("7*x^2");

  await answer.fill("14*x");
  await answer.press("Enter");
  await expect(page.getByText("Update 6")).toBeVisible();
  await expect(page.locator(".assessment-bubble").last()).toContainText("x^(-2)");

  await answer.fill("-2*x^(-3)");
  await answer.press("Enter");
  await expect(page.getByText("Update 7")).toBeVisible();
  await expect(
    page.getByText("Differentiate the whole expression:", { exact: true }),
  ).toBeVisible();

  await answer.fill("35*x^6 - 6*x^2");
  await answer.press("Enter");
  await expect(page.getByText("Update 8")).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Learning path complete" }),
  ).toBeVisible();
  await expect(
    page
      .getByRole("region", { name: "Tutor conversation history" })
      .getByText(/was solved independently/i),
  ).toBeVisible();
  await expectNoAccessibilityViolations(page, "completed lesson");
});

test("a competing tab receives the authoritative stale snapshot", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Start my learning path" }).click();
  await expect(page.getByText("Update 0")).toBeVisible();

  const competingTab = await page.context().newPage();
  await competingTab.goto("/");
  await expect(competingTab.getByText("Update 0")).toBeVisible();

  await page.getByRole("button", { name: "Get a hint" }).click();
  await expect(page.getByText("Update 1")).toBeVisible();

  const competingAnswer = competingTab.getByLabel("Your expression");
  await competingAnswer.fill("0");
  await competingAnswer.press("Enter");
  await expect(competingTab.getByText("Update 1")).toBeVisible();
  await expect(competingTab.getByRole("alert")).toBeVisible();
  await expect(
    competingTab.getByText(
      /revision changed|authoritative snapshot|changed elsewhere|latest state/i,
    ),
  ).toBeVisible();
  await expect(page.getByText("Update 1")).toBeVisible();
});

test("reload restores the exact authoritative pending interaction", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Start my learning path" }).click();
  const answer = page.getByLabel("Your expression");
  await answer.fill("0");
  await answer.press("Enter");
  await expect(page.getByText("Update 1")).toBeVisible();
  const pendingPrompt = await page.locator(".assessment-bubble").last().textContent();

  await page.reload();

  await expect(page.getByText("Your session was restored.")).toBeVisible();
  await expect(page.getByText("Update 1")).toBeVisible();
  await expect(page.locator(".assessment-bubble").last()).toContainText(
    pendingPrompt.trim(),
  );
  await expect(answer).toBeFocused();
  await expectNoAccessibilityViolations(page, "restored interaction");
});

test("reload recovers a committed create whose response and cookie were lost", async ({
  page,
}) => {
  await page.goto("/");
  await page.route("**/api/v2/sessions", async (route) => {
    await route.fetch();
    await route.abort("failed");
  });

  await page.getByRole("button", { name: "Start my learning path" }).click();
  await expect(page.getByRole("alert")).toContainText(/could not be reached/i);
  const stored = await page.evaluate(() =>
    sessionStorage.getItem("tutor.v2.pending-recovery.v1"),
  );
  expect(JSON.parse(stored)).toMatchObject({
    schema_version: 1,
    operation: "create",
  });
  await page.unroute("**/api/v2/sessions");

  await page.reload();

  await expect(page.getByText("Update 0")).toBeVisible();
  await expect(page.getByLabel("Your expression")).toBeFocused();
  await expect
    .poll(() =>
      page.evaluate(() =>
        sessionStorage.getItem("tutor.v2.pending-recovery.v1"),
      ),
    )
    .toBeNull();
});

test("reload recovers a committed reset only with its request proof", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Start my learning path" }).click();
  await expect(page.getByText("Update 0")).toBeVisible();
  await page.route("**/api/v2/sessions/current/reset", async (route) => {
    await route.fetch();
    await route.abort("failed");
  });

  await page.getByRole("button", { name: "Restart this goal" }).click();
  await expect(page.getByRole("alert")).toContainText(/could not be reached/i);
  const stored = await page.evaluate(() =>
    sessionStorage.getItem("tutor.v2.pending-recovery.v1"),
  );
  expect(JSON.parse(stored)).toMatchObject({
    schema_version: 1,
    operation: "reset",
  });
  await page.unroute("**/api/v2/sessions/current/reset");

  await page.reload();

  await expect(page.getByText("Update 0")).toBeVisible();
  await expect(page.getByLabel("Your expression")).toBeFocused();
  await expect
    .poll(() =>
      page.evaluate(() =>
        sessionStorage.getItem("tutor.v2.pending-recovery.v1"),
      ),
    )
    .toBeNull();
});

test("a malformed archived widget fails safely without replacing the active input", async ({
  page,
}) => {
  await page.route("**/api/v2/sessions", async (route) => {
    const response = await route.fetch();
    const snapshot = await response.json();
    snapshot.transcript.push({
      sequence: snapshot.transcript.length,
      role: "tutor",
      kind: "lesson",
      text: "Guided visual practice",
      interaction_key: "archived-malformed-widget",
      kc_id: snapshot.pending.kc_id,
      widget: {
        widget_type: "slider",
        learning_objective: "Safely recover from invalid render data",
        prompt: "Use the accessible alternative.",
        text_fallback: "Continue with text practice.",
        params: null,
      },
    });
    await route.fulfill({ response, json: snapshot });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "Start my learning path" }).click();

  await expect(
    page.getByText("Accessible alternative needed", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText(/could not be initialized.*text alternative/i),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Continue with a text alternative" }),
  ).toHaveCount(0);
  await expect(page.getByLabel("Your expression")).toBeFocused();
  await expectNoAccessibilityViolations(page, "malformed widget fallback");
});
