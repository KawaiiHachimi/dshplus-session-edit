import { randomUUID } from 'node:crypto';
import { open, rename, rm } from 'node:fs/promises';
import { isAbsolute } from 'node:path';
import { Session } from '@deepseek-ai/dsh-session';
import { createUserMessage } from '@deepseek-ai/dsh-llm';
import { isHuman, rewriteSeed } from './plan.js';
import { fail, SessionEditError } from './errors.js';

export function historyUpdatedFailure(error) {
  return new SessionEditError(
    'regeneration-failed',
    'The edited history was updated, but regeneration could not be confirmed. Open this session to check its state.',
    { historyUpdated: true },
    { cause: error },
  );
}

// The tested DSH runtime exposes its complete, ordered agent teardown as a
// labeled Cordis effect. Calling it drains the driver, closes the writer and
// unregisters the agent/session; disposing just agent.ctx would miss those steps.
// The effect lives on the fiber of whichever Context created the Agent, so every
// loaded fiber and its ancestors are searched; Cordis 4.2 gives the root Context
// no fiber of its own.
function candidateFibers(ctx) {
  const seen = new Set();
  const fibers = [];
  const include = start => {
    let fiber = start;
    while (fiber && !seen.has(fiber)) {
      seen.add(fiber);
      fibers.push(fiber);
      fiber = fiber.parent?.fiber;
    }
  };
  include(ctx.fiber);
  for (const runtime of ctx.registry?.values?.() ?? []) {
    for (const fiber of runtime?.fibers ?? []) include(fiber);
  }
  return fibers;
}

function teardownFor(ctx, agent) {
  const label = 'agentLoop.lifecycle(' + agent.id + ')';
  try {
    for (const fiber of candidateFibers(ctx)) {
      const dispose = [...(fiber._disposables ?? [])].find(
        effect => typeof effect === 'function' && effect[Symbol.for('cordis.effect')]?.label === label,
      );
      if (dispose) return dispose;
    }
  } catch {
    // A changed Cordis effect layout only makes history rewriting unavailable.
  }
  fail('unsupported-runtime', 'This agent runtime does not support history regeneration.');
}

export async function rewriteHistory(ctx, snapshot, plan, now) {
  // rewriteSeed emits v4 events. Reading and native continuation do not rewrite
  // that representation, so a format change must not prevent plugin startup.
  if (snapshot.header.version !== 4) {
    fail('unsupported-format', 'History regeneration requires DSH session format v4.');
  }
  const persistence = ctx.sessionPersistence;
  const cache = ctx.get?.('sessionProjectionCache');
  // Detect the backend by the capability this rewrite needs. `persistence.name`
  // is the Cordis service key ('sessionPersistence'), and DSH 0.2.3 dropped the
  // per-class name override that used to shadow it.
  if (
    typeof persistence.open !== 'function' ||
    typeof persistence.encodeMaterialization !== 'function' ||
    typeof persistence.locate !== 'function' ||
    (cache && typeof cache.table?.delete !== 'function')
  ) fail('unsupported-backend', 'History regeneration requires DSH JSONL session storage.');

  const live = ctx.agents.get(plan.sessionId);
  const dispose = live ? teardownFor(ctx, live) : undefined;
  let retired = false;
  let committed = false;
  let handle;
  let temporary;
  let failure;
  let result;
  try {
    if (live) {
      if (live.status === 'running') {
        live.cancel({ kind: 'user' });
        await live.whenIdle();
      }
      retired = true;
      await dispose();
    }
    handle = await persistence.open(plan.sessionId, 'write');
    const { events } = await handle.read();
    const target = events[plan.eventSeq];
    if (!isHuman(target ?? {}) || target.data.id !== plan.messageId) {
      fail('session-changed', 'The selected message changed. Reopen it and try again.');
    }
    const message = {
      ...target.data,
      id: createUserMessage({ content: plan.content, source: target.data.source }).id,
      content: plan.content,
    };
    const seed = rewriteSeed({
      ...snapshot, header: handle.header, events, inheritedEventCount: handle.inheritedEventCount,
    }, plan, message, now);
    result = { editedSeq: seed.editedSeq };
    Session.create(plan.sessionId, seed.events, handle.header, seed.inheritedEventCount, ctx.sessions.messageProjections);
    const location = persistence.locate(handle.header);
    if (location.kind !== 'jsonl' || !isAbsolute(location.path)) {
      fail('unsupported-backend', 'History regeneration requires local JSONL session storage.');
    }
    const bytes = await persistence.encodeMaterialization(handle.header, seed.inheritedEventCount, seed.events);
    temporary = location.path + '.session-edit-' + randomUUID() + '.tmp';
    const file = await open(temporary, 'wx', 0o600);
    try {
      await file.writeFile(bytes);
      await file.sync();
    } finally {
      await file.close();
    }
    // A projection checkpoint can have a higher watermark than the rewritten
    // log. Drop it after the old lifecycle's final writes have drained.
    await cache?.table.delete(plan.sessionId);
    if (ctx.sessionPersistence.identity !== persistence.identity || ctx.sessions.get(plan.sessionId)) {
      fail('session-changed', 'The session was activated again before its history could be replaced.');
    }
    await rename(temporary, location.path);
    committed = true;
    temporary = undefined;
  } catch (error) {
    failure = error;
  } finally {
    if (temporary) await rm(temporary, { force: true }).catch(() => {});
    try { await handle?.close(); } catch (error) { failure ??= error; }
  }
  if (failure) {
    // A failed pre-commit attempt still has its original durable history.
    // Republish it so native clients do not remain in the removed state.
    if (retired && !committed) await ctx.sessionController.resolveAgent(plan.sessionId).catch(() => {});
    throw committed ? historyUpdatedFailure(failure) : failure;
  }
  return result;
}
