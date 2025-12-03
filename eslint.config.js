// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export default [
  {
    files: ["**/*.js", "**/*.ts"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
    },
    rules: {
      // Possible errors
      "no-console": "off",
      "no-debugger": "warn",
      "no-undef": "off",  // no global checking
      "no-unused-vars": "off",  // turned off since unused exports may be used by other files
      
      // Best practices
      "eqeqeq": ["error", "always"],  // changed from error to warn
      "no-var": "error",  // changed from error to warn
      "prefer-const": "error",  // turned off for more flexibility
      
      // Stylistic
      "semi": ["error", "never"],  // Enforce no semicolons
      "quotes": ["error", "double", { avoidEscape: true }],  // Enforce double quotes
    },
  },
]

