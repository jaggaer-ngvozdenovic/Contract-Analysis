import fromPolyfill from 'core-js/library/fn/array/from';
import isArrayPolyfill from 'core-js/library/fn/array/is-array'; // --- Static ---

export var from = Array.from || fromPolyfill;
export var isArray = Array.isArray || isArrayPolyfill; // --- Instance ---

export var arrayIncludes = function arrayIncludes(array, value) {
  return array.indexOf(value) !== -1;
};
export var concat = function concat() {
  for (var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++) {
    args[_key] = arguments[_key];
  }

  return Array.prototype.concat.apply([], args);
};