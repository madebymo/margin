import TutorScene from "../elm/TutorScene.elm";

const EMPTY_STATE = {
  scene: null,
  status: "Rich visual unavailable.",
};

class TutorSceneElement extends HTMLElement {
  #app = null;
  #sceneState = EMPTY_STATE;
  #forwardInteraction = (detail) => {
    this.dispatchEvent(
      new CustomEvent("interact", {
        bubbles: true,
        composed: true,
        detail,
      }),
    );
  };

  connectedCallback() {
    if (this.#app) return;

    const mountPoint = document.createElement("div");
    this.replaceChildren(mountPoint);
    this.#app = TutorScene.init({
      node: mountPoint,
      flags: this.#sceneState,
    });
    this.#app.ports.interact.subscribe(this.#forwardInteraction);
  }

  disconnectedCallback() {
    if (!this.#app) return;
    const app = this.#app;
    this.#app = null;
    app.ports.interact.unsubscribe(this.#forwardInteraction);
    if (typeof app.unmount === "function") {
      app.unmount();
    }
  }

  set sceneState(value) {
    this.#sceneState = value ?? EMPTY_STATE;
    if (this.#app) {
      this.#app.ports.sceneStateIn.send(this.#sceneState);
    }
  }

  get sceneState() {
    return this.#sceneState;
  }
}

if (!customElements.get("tutor-scene")) {
  customElements.define("tutor-scene", TutorSceneElement);
}
